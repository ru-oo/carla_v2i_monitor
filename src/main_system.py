"""
main_system.py
==============
담당: 전체 팀 (최종 통합 실행 파일)
브랜치: feature/cnn-model

역할:
    - CARLA V2I 스마트 교차로 3D 관제 시스템 메인 루프
    - 항공 조감 CCTV (50m 수직 + 교차로 전체 시야)
    - Occlusion Culling (cast_ray) 로 건물 뒤 차량 필터링
    - Open3D 3D V2I 레이더 맵 (Tesla 계기판 스타일)
    - VisionProcessor (MOG2 + CentroidTracker) 디버그 패널

실행:
    python src/main_system.py
조작:
    q — 종료  |  r — NPC 재소환
"""

import math
import os
import queue
import random
import sys
import time
import traceback

import carla
import cv2
import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classifier import CLASSES, VehicleClassifier
from vision_processor import VisionProcessor


# =============================================
# 설정값
# =============================================
CARLA_HOST = "localhost"
CARLA_PORT = 2000
TM_PORT    = 9000

IMG_W      = 1280
IMG_H      = 720
CAM_FOV    = 90.0       # 넓은 FOV → 교차로 전체 포착
CAM_HEIGHT = 50.0       # 항공 조감 높이 (m) — 교차로 전체 보임
NUM_NPC    = 40

MAP_PX     = 500        # Open3D 레이더 캔버스 크기

CLASSIFY_INTERVAL      = 5
MIN_BBOX_PX            = 8    # 항공뷰: 작은 BBox도 유효
RADAR_UPDATE_INTERVAL  = 6    # Open3D 갱신 주기 (프레임)
OCCLUSION_CACHE_FRAMES = 10
MAX_DETECTION_DIST     = 90.0
VISION_UPDATE_INTERVAL = 3

_3D_ROAD_LEN = 55.0    # 도로 암 길이 (m)

# ─── 차종 색상 (BGR) ────
COLORS = {
    "bus":     (0,   140, 255),
    "car":     (0,   210,  60),
    "truck":   (50,   50, 255),
    "van":     (255, 180,   0),
    "unknown": (150, 150, 150),
}
ICON = {"bus": "B", "car": "C", "truck": "T", "van": "V", "unknown": "?"}

# ─── Open3D용 RGB [0,1] ────
_O3D_RGB = {
    "bus":     (1.00, 0.55, 0.00),
    "car":     (0.00, 0.82, 0.24),
    "truck":   (0.20, 0.20, 1.00),
    "van":     (1.00, 0.71, 0.00),
    "unknown": (0.59, 0.59, 0.59),
}
_HEX = {
    "bus": "#FF8C00", "car": "#00D23C",
    "truck": "#3232FF", "van": "#FFB400", "unknown": "#969696",
}


# =============================================
# 유틸: 카메라 투영
# =============================================
def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * math.tan(math.radians(fov) / 2.0))
    return np.array([[focal, 0, w/2.0], [0, focal, h/2.0], [0, 0, 1.0]])


def world_to_pixel(world_loc, camera_actor, K):
    cam_inv = np.array(camera_actor.get_transform().get_inverse_matrix())
    pt      = cam_inv @ np.array([world_loc.x, world_loc.y, world_loc.z, 1.0])
    x_img, y_img, z_img = pt[1], -pt[2], pt[0]
    if z_img <= 0:
        return None
    u = int(K[0, 0] * (x_img / z_img) + K[0, 2])
    v = int(K[1, 1] * (y_img / z_img) + K[1, 2])
    return u, v


def get_vehicle_bbox_pixels(vehicle, camera_actor, K, img_w, img_h):
    verts  = vehicle.bounding_box.get_world_vertices(vehicle.get_transform())
    us, vs = [], []
    for vert in verts:
        px = world_to_pixel(vert, camera_actor, K)
        if px is not None:
            us.append(px[0]); vs.append(px[1])
    if len(us) < 2:
        return None
    x1, y1 = min(us), min(vs)
    x2, y2 = max(us), max(vs)
    if x2 < 0 or y2 < 0 or x1 >= img_w or y1 >= img_h:
        return None
    return (max(x1,0), max(y1,0), min(x2,img_w-1), min(y2,img_h-1))


# =============================================
# Occlusion Culling
# =============================================
def is_vehicle_visible(world, cam_loc, vehicle, max_dist=MAX_DETECTION_DIST):
    veh_loc  = vehicle.get_location()
    veh_dist = cam_loc.distance(veh_loc)
    if veh_dist > max_dist:
        return False
    try:
        hits = world.cast_ray(cam_loc, veh_loc)
    except Exception:
        return True
    if not hits:
        return True
    return cam_loc.distance(hits[0].location) >= veh_dist * 0.85


# =============================================
# 도로 반폭 측정
# =============================================
def get_road_half_width(world, junction_center, default=6.5):
    try:
        carla_map = world.get_map()
        widths = []
        for dx, dy in [(15,0),(-15,0),(0,15),(0,-15)]:
            loc = carla.Location(
                x=junction_center.x+dx, y=junction_center.y+dy,
                z=junction_center.z)
            wp = carla_map.get_waypoint(loc, project_to_road=True)
            if wp is None:
                continue
            total = wp.lane_width
            rw = wp
            for _ in range(6):
                nxt = rw.get_right_lane()
                if nxt is None or nxt.lane_type != carla.LaneType.Driving: break
                total += nxt.lane_width; rw = nxt
            lw = wp
            for _ in range(6):
                nxt = lw.get_left_lane()
                if nxt is None or nxt.lane_type != carla.LaneType.Driving: break
                total += nxt.lane_width; lw = nxt
            widths.append(total / 2.0)
        if widths:
            hw = max(3.5, min(sum(widths)/len(widths), 18.0))
            print(f"[V2I] 도로 반폭: {hw:.2f}m")
            return hw
    except Exception as e:
        print(f"[V2I] 도로폭 측정 실패: {e}")
    return default


# =============================================
# 교차로 탐색
# =============================================
def find_intersection(world):
    carla_map = world.get_map()
    junc_wps  = [wp for wp in carla_map.generate_waypoints(5.0) if wp.is_junction]
    if not junc_wps:
        sps = carla_map.get_spawn_points()
        cx  = sum(s.location.x for s in sps)/len(sps)
        cy  = sum(s.location.y for s in sps)/len(sps)
        return carla.Location(x=cx, y=cy, z=0.0)
    cx = sum(wp.transform.location.x for wp in junc_wps)/len(junc_wps)
    cy = sum(wp.transform.location.y for wp in junc_wps)/len(junc_wps)
    best = min(junc_wps,
               key=lambda wp: (wp.transform.location.x-cx)**2
                            + (wp.transform.location.y-cy)**2)
    loc  = best.transform.location
    print(f"[V2I] 교차로: ({loc.x:.1f}, {loc.y:.1f})")
    return carla.Location(x=loc.x, y=loc.y, z=loc.z)


# =============================================
# 항공 CCTV 카메라 설치 (교차로 바로 위)
# =============================================
def spawn_cctv_camera(world, junc_center):
    """
    교차로 중심 정 상공 CAM_HEIGHT(50m)에 카메라 설치.
    pitch=-89°로 거의 수직 내려다보기 → 교차로 전체가 한 눈에 보임.
    """
    bp_lib = world.get_blueprint_library()
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(IMG_W))
    cam_bp.set_attribute("image_size_y", str(IMG_H))
    cam_bp.set_attribute("fov", str(CAM_FOV))

    cx = junc_center.x
    cy = junc_center.y
    cz = junc_center.z + CAM_HEIGHT   # 정 상공

    pitch = -89.0   # 거의 수직 내려다보기 (짐벌 락 방지)
    yaw   = -90.0   # 북쪽을 화면 위로 → 지도 방향과 일치

    cam_tf = carla.Transform(
        carla.Location(x=cx, y=cy, z=cz),
        carla.Rotation(pitch=pitch, yaw=yaw, roll=0.0),
    )

    img_q  = queue.Queue(maxsize=2)
    camera = world.spawn_actor(cam_bp, cam_tf)
    camera.listen(lambda img: img_q.put_nowait(img) if not img_q.full() else None)

    # 화면 중심에서 보이는 지면 범위 ≈ 2 * 50 * tan(45°) = 100m
    print(f"[V2I] 항공 CCTV  ({cx:.1f}, {cy:.1f}, {cz:.1f}m)  "
          f"pitch={pitch}°  yaw={yaw}°  시야≈100m")
    return camera, img_q, cam_tf


# =============================================
# NPC 소환
# =============================================
def spawn_npc_vehicles(client, world, tm, num=NUM_NPC):
    bp_lib    = world.get_blueprint_library()
    all_bps   = [bp for bp in bp_lib.filter("vehicle.*")
                 if int(bp.get_attribute("number_of_wheels")) >= 4]
    spawn_pts = world.get_map().get_spawn_points()
    random.shuffle(spawn_pts)
    batch = []
    for sp in spawn_pts[:num]:
        bp = random.choice(all_bps)
        batch.append(
            carla.command.SpawnActor(bp, sp).then(
                carla.command.SetAutopilot(
                    carla.command.FutureActor, True, tm.get_port())
            )
        )
    vehicles = []
    for r in client.apply_batch_sync(batch, True):
        if not r.error:
            actor = world.get_actor(r.actor_id)
            if actor:
                tm.distance_to_leading_vehicle(actor, random.uniform(2.0, 4.0))
                tm.vehicle_percentage_speed_difference(actor, random.uniform(-10, 10))
                tm.auto_lane_change(actor, False)
                vehicles.append(actor)
    print(f"[V2I] NPC 소환: {len(vehicles)}대")
    return vehicles


# =============================================
# Open3D 3D V2I 레이더 맵
# =============================================
class V2IMapRenderer:
    """
    Open3D Visualizer 기반 Tesla 계기판 스타일 3D V2I 레이더

    ┌─ 설계 ─────────────────────────────────────────────────────┐
    │  오픈 소스 3D 렌더러 (Open3D 0.19)                         │
    │  - TriangleMesh: 차량 박스 (상면 밝게, 측면 중간, 하면 어둠) │
    │  - LineSet: V2I 신호선, 거리 원호, 차선, CCTV 폴            │
    │  - 정적 장면은 __init__에서 한 번만 생성                    │
    │  - 동적 장면(차량, V2I선)은 매 RADAR_UPDATE_INTERVAL마다   │
    │    remove → add 방식으로 갱신                               │
    │  - 렌더 이미지: capture_screen_float_buffer → BGR numpy      │
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(self, junction_center: carla.Location,
                 cam_location: carla.Location,
                 road_half_width: float = 6.5):
        self.junc    = junction_center
        self.cam_loc = cam_location
        self.road_hw = road_half_width

        # ── Open3D Visualizer (비가시 윈도우) ──────────────────
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(
            window_name="V2I Radar (Open3D)",
            width=MAP_PX, height=MAP_PX,
            visible=False,
        )

        # 렌더 옵션
        opt = self.vis.get_render_option()
        opt.background_color = np.array([0.04, 0.047, 0.071])  # Tesla 어두운 배경
        opt.point_size       = 4.0
        opt.line_width       = 2.5
        opt.light_on         = False    # Unlit 렌더 → 색상 그대로 표현

        # 정적 장면 구축
        self._static_geoms: list = []
        self._build_static_scene()

        # 초기 렌더 (bounding box 확립)
        self.vis.poll_events()
        self.vis.update_renderer()

        # 카메라 고정
        self._apply_camera()

        # 동적 장면 추적
        self._dyn_geoms: list = []

        self._last_render  = np.zeros((MAP_PX, MAP_PX, 3), dtype=np.uint8)
        self._render_count = 0

        print(f"[V2IMapRenderer] Open3D 초기화 완료  도로반폭={road_half_width:.1f}m")

    # ── 좌표 변환 ──────────────────────────────────────────────
    def _c(self, cx, cy, cz=0.0):
        """CARLA world → 장면 로컬 좌표 (교차로 중심, Y=South 유지)"""
        return (float(cx - self.junc.x),
                float(cy - self.junc.y),
                float(cz - self.junc.z))

    # ── 유틸: 사각형 슬래브 메쉬 ─────────────────────────────
    @staticmethod
    def _flat_rect(x1, y1, x2, y2, z=0.0, thick=0.08):
        dx, dy = x2-x1, y2-y1
        box    = o3d.geometry.TriangleMesh.create_box(
            width=abs(dx), height=abs(dy), depth=thick
        )
        box.translate([min(x1,x2), min(y1,y2), z - thick])
        return box

    @staticmethod
    def _rot_z(angle_rad):
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=float)

    # ── 정적 장면 구축 ────────────────────────────────────────
    def _add(self, geom, color_rgb=None, static=True):
        """add_geometry 래퍼. color_rgb=(r,g,b) 이면 uniform color 적용"""
        if color_rgb is not None:
            geom.paint_uniform_color(color_rgb)
        if hasattr(geom, "compute_vertex_normals"):
            geom.compute_vertex_normals()
        self.vis.add_geometry(geom, reset_bounding_box=static)
        if static:
            self._static_geoms.append(geom)
        else:
            self._dyn_geoms.append(geom)

    def _build_static_scene(self):
        hw = self.road_hw
        ln = _3D_ROAD_LEN

        ROAD_COL   = [0.15, 0.17, 0.21]
        GROUND_COL = [0.04, 0.047, 0.071]
        ARC_COL    = [0.11, 0.15, 0.22]
        LANE_COL   = [0.24, 0.28, 0.37]
        CCTV_COL   = [0.0,  0.82, 0.54]
        CENTER_COL = [0.0,  0.78, 1.0 ]

        # 지면
        gnd = self._flat_rect(-65, -65, 65, 65, z=0.0, thick=0.05)
        self._add(gnd, GROUND_COL)

        # 도로 암 (North=-Y, South=+Y, East=+X, West=-X)
        road_segs = [
            (-hw, -ln, hw,  0),    # 북 (−Y 방향)
            (-hw,  0,  hw, ln),    # 남 (+Y 방향)
            ( 0, -hw, ln,  hw),    # 동 (+X 방향)
            (-ln, -hw,  0, hw),    # 서 (−X 방향)
            (-hw, -hw, hw, hw),    # 교차로 중앙
        ]
        for i, (x1,y1,x2,y2) in enumerate(road_segs):
            r = self._flat_rect(x1, y1, x2, y2, z=0.01, thick=0.06)
            self._add(r, ROAD_COL)

        # 차선 중앙선 (LineSet)
        lane_pts, lane_lines, lane_cols = [], [], []
        seg, gap = 2.2, 1.4
        for ddx, ddy in ((0,-1),(0,1),(1,0),(-1,0)):
            k = 1
            while True:
                t0 = k*(seg+gap) - seg
                t1 = t0 + seg
                if t1 > ln: break
                i0 = len(lane_pts)
                lane_pts += [[ddx*t0, ddy*t0, 0.03], [ddx*t1, ddy*t1, 0.03]]
                lane_lines.append([i0, i0+1])
                lane_cols.append(LANE_COL)
                k += 1
        if lane_lines:
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(lane_pts)
            ls.lines  = o3d.utility.Vector2iVector(lane_lines)
            ls.colors = o3d.utility.Vector3dVector(lane_cols)
            self.vis.add_geometry(ls, reset_bounding_box=True)
            self._static_geoms.append(ls)

        # 거리 원호 (10/20/30/40m)
        for r_m in (10, 20, 30, 40):
            n       = 72
            thetas  = np.linspace(0, 2*np.pi, n, endpoint=False)
            pts     = [[r_m*np.cos(t), r_m*np.sin(t), 0.04] for t in thetas]
            lines   = [[i, (i+1)%n] for i in range(n)]
            cols    = [ARC_COL] * n
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(pts)
            ls.lines  = o3d.utility.Vector2iVector(lines)
            ls.colors = o3d.utility.Vector3dVector(cols)
            self.vis.add_geometry(ls, reset_bounding_box=True)
            self._static_geoms.append(ls)

        # 교차로 중심 마커 (작은 원기둥)
        cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=0.8, height=0.6)
        cyl.translate([0, 0, 0.3])
        self._add(cyl, CENTER_COL)

        # CCTV 폴 (교차로 정 상공 → 세로 라인)
        cx, cy, cz = self._c(self.cam_loc.x, self.cam_loc.y, self.cam_loc.z)
        pole = o3d.geometry.LineSet()
        pole.points = o3d.utility.Vector3dVector([[cx,cy,0],[cx,cy,cz]])
        pole.lines  = o3d.utility.Vector2iVector([[0,1]])
        pole.colors = o3d.utility.Vector3dVector([CCTV_COL])
        self.vis.add_geometry(pole, reset_bounding_box=True)
        self._static_geoms.append(pole)

        # CCTV 카메라 마커 (작은 구)
        ball = o3d.geometry.TriangleMesh.create_sphere(radius=1.2)
        ball.translate([cx, cy, cz])
        self._add(ball, CCTV_COL)

    # ── 카메라 뷰 고정 (Tesla 계기판 느낌) ──────────────────────
    def _apply_camera(self):
        """
        교차로 남쪽 + 약간 동쪽, 높이 30m → 북서 방향 내려다봄
        (CARLA Y=South → 남쪽에서 북쪽 방향으로 바라보는 Tesla 계기판 시점)
        """
        ctr = self.vis.get_view_control()
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, 0, 1])
        # front: 교차로를 향하는 방향 = 남쪽(+Y)과 위쪽(+Z)에서 바라봄
        # 정면 벡터 = eye→target = 0-(10,40,28) normalize
        ctr.set_front([-0.18, -0.78, -0.60])
        ctr.set_zoom(0.28)

    # ── 동적 장면 정리 ────────────────────────────────────────
    def _clear_dynamic(self):
        for geom in self._dyn_geoms:
            try:
                self.vis.remove_geometry(geom, reset_bounding_box=False)
            except Exception:
                pass
        self._dyn_geoms.clear()

    # ── 차량 박스 생성 ────────────────────────────────────────
    def _make_vehicle_box(self, actor, vtype: str):
        loc = actor.get_location()
        px, py, pz = self._c(loc.x, loc.y, loc.z)

        bb = actor.bounding_box
        hl, hw_v, hh = bb.extent.x, bb.extent.y, bb.extent.z
        yaw_rad = math.radians(actor.get_transform().rotation.yaw)

        # 박스 생성 (중심 원점, Z=0이 바닥)
        box = o3d.geometry.TriangleMesh.create_box(
            width=hl*2, height=hw_v*2, depth=hh*2
        )
        box.translate([-hl, -hw_v, 0.0])   # XY 중앙 정렬

        # 수직 높이에 따른 퍼 버텍스 색상 (하=어두움, 상=밝음)
        verts = np.asarray(box.vertices)
        base  = np.array(_O3D_RGB.get(vtype, _O3D_RGB["unknown"]))
        colors_arr = np.zeros((len(verts), 3))
        for i, v in enumerate(verts):
            z_frac         = np.clip(v[2] / (hh*2), 0, 1)
            brightness     = 0.22 + 0.78 * z_frac
            colors_arr[i]  = np.clip(base * brightness, 0, 1)
        box.vertex_colors = o3d.utility.Vector3dVector(colors_arr)
        box.compute_vertex_normals()

        # Z축 회전 (CARLA yaw = Open3D Z rotation 동방향)
        R = self._rot_z(yaw_rad)
        box.rotate(R, center=[0, 0, 0])
        box.translate([px, py, pz])
        return box

    # ── V2I 신호선 (LineSet) ─────────────────────────────────
    def _make_v2i_lineset(self, results: list):
        pts   = [[0.0, 0.0, 0.5]]
        lines = []
        cols  = []
        for r in results:
            loc = r["actor"].get_location()
            px, py, pz = self._c(loc.x, loc.y, loc.z)
            if abs(px) > 53 or abs(py) > 53:
                continue
            idx = len(pts)
            pts.append([px, py, max(pz, 0.5)])
            lines.append([0, idx])
            base = _O3D_RGB.get(r["type"], _O3D_RGB["unknown"])
            cols.append([base[0]*0.5, base[1]*0.5, base[2]*0.5])
        if not lines:
            return None
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(pts)
        ls.lines  = o3d.utility.Vector2iVector(lines)
        ls.colors = o3d.utility.Vector3dVector(cols)
        return ls

    # ── 범례 오버레이 (cv2) ──────────────────────────────────
    @staticmethod
    def _draw_legend(img: np.ndarray, counts: dict, total: int) -> np.ndarray:
        """Open3D 렌더 이미지 위에 cv2로 범례 + 타이틀 오버레이"""
        # 타이틀
        cv2.putText(img, "3D V2I RADAR", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "3D V2I RADAR", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 220, 255), 1, cv2.LINE_AA)

        # 범례 패널 (우하단 반투명)
        pw, ph = 130, len(CLASSES)*22 + 32
        px0    = MAP_PX - pw - 6
        py0    = MAP_PX - ph - 6
        roi    = img[py0:py0+ph, px0:px0+pw].copy()
        cv2.addWeighted(roi, 0.25, np.zeros_like(roi), 0.75, 0, roi)
        img[py0:py0+ph, px0:px0+pw] = roi
        cv2.rectangle(img, (px0, py0), (px0+pw, py0+ph), (30,40,60), 1)

        cv2.putText(img, "V2I RADAR", (px0+6, py0+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,180,255), 1)
        for i, cls in enumerate(CLASSES):
            color = COLORS[cls]
            iy    = py0 + 22 + i*22
            cv2.rectangle(img, (px0+6, iy-7), (px0+18, iy+5), color, -1)
            cv2.putText(img, f"{cls.upper():<5} {counts.get(cls,0)}",
                        (px0+22, iy+4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1)
        cv2.putText(img, f"TOTAL  {total}",
                    (px0+6, py0+ph-7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (200,200,200), 1)
        return img

    # ── 메인 렌더 ────────────────────────────────────────────
    def render(self, results: list) -> np.ndarray:
        self._render_count += 1
        if self._render_count % RADAR_UPDATE_INTERVAL != 0:
            return self._last_render

        # ── 동적 장면 갱신 ──────────────────────────────────
        self._clear_dynamic()

        # 차량 박스
        for r in results:
            loc = r["actor"].get_location()
            px, py = self._c(loc.x, loc.y)[0:2]
            if abs(px) > 53 or abs(py) > 53:
                continue
            box = self._make_vehicle_box(r["actor"], r["type"])
            self.vis.add_geometry(box, reset_bounding_box=False)
            self._dyn_geoms.append(box)

        # V2I 신호선
        v2i_ls = self._make_v2i_lineset(results)
        if v2i_ls is not None:
            self.vis.add_geometry(v2i_ls, reset_bounding_box=False)
            self._dyn_geoms.append(v2i_ls)

        # ── 렌더 + 이미지 캡처 ──────────────────────────────
        self._apply_camera()        # 카메라 드리프트 방지
        self.vis.poll_events()
        self.vis.update_renderer()
        img_float = self.vis.capture_screen_float_buffer(do_render=True)
        img       = (np.asarray(img_float) * 255).astype(np.uint8)
        bgr       = img[:, :, 2::-1].copy()   # RGB → BGR

        if bgr.shape[:2] != (MAP_PX, MAP_PX):
            bgr = cv2.resize(bgr, (MAP_PX, MAP_PX))

        # ── cv2 오버레이 (범례·거리 라벨) ─────────────────────
        counts = {c: 0 for c in CLASSES}
        for r in results:
            if r["type"] in counts:
                counts[r["type"]] += 1
        bgr = self._draw_legend(bgr, counts, len(results))

        self._last_render = bgr
        return bgr

    def destroy(self):
        try:
            self.vis.destroy_window()
        except Exception:
            pass


# =============================================
# V2I 관제 시스템
# =============================================
class V2IMonitorSystem:
    def __init__(self):
        self.client = carla.Client(CARLA_HOST, CARLA_PORT)
        self.client.set_timeout(15.0)
        self.world  = self.client.get_world()
        self._map_name = self.world.get_map().name.split("/")[-1]
        print(f"[V2I] CARLA 연결  맵: {self._map_name}")

        settings = self.world.get_settings()
        settings.synchronous_mode    = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

        self.tm = self.client.get_trafficmanager(TM_PORT)
        self.tm.set_synchronous_mode(True)
        self.tm.set_global_distance_to_leading_vehicle(2.5)

        self.junction_center = find_intersection(self.world)
        road_hw = get_road_half_width(self.world, self.junction_center)

        self.camera, self.img_queue, self.cam_tf = spawn_cctv_camera(
            self.world, self.junction_center
        )
        self.K = build_projection_matrix(IMG_W, IMG_H, CAM_FOV)

        for _ in range(10):
            self.world.tick()
        self.npc_list = spawn_npc_vehicles(self.client, self.world, self.tm, NUM_NPC)

        print("[V2I] 워밍업 중... (60 tick)")
        for _ in range(60):
            self.world.tick()
        while not self.img_queue.empty():
            try:   self.img_queue.get_nowait()
            except queue.Empty: break

        self.classifier          = VehicleClassifier()
        self._classify_cache: dict[int, str]            = {}
        self._occlusion_cache: dict[int, tuple[bool,int]] = {}
        self._frame_idx          = 0

        self.vision_proc = VisionProcessor()

        self.radar = V2IMapRenderer(
            junction_center=self.junction_center,
            cam_location=self.cam_tf.location,
            road_half_width=road_hw,
        )

        self._t_last = time.time()
        self._fps    = 0.0

        print("[V2I] 초기화 완료  q=종료  r=NPC재소환")

    # ─────────────────────────────────────────────────────────
    def _is_occluded(self, vehicle) -> bool:
        vid    = vehicle.id
        cached = self._occlusion_cache.get(vid)
        if cached is not None:
            visible, last_f = cached
            if self._frame_idx - last_f < OCCLUSION_CACHE_FRAMES:
                return not visible
        visible = is_vehicle_visible(self.world, self.cam_tf.location, vehicle)
        self._occlusion_cache[vid] = (visible, self._frame_idx)
        return not visible

    # ─────────────────────────────────────────────────────────
    def _detect_and_classify(self, frame: np.ndarray) -> list:
        self._frame_idx += 1
        results = []
        for vehicle in self.world.get_actors().filter("vehicle.*"):
            vid = vehicle.id

            if self._is_occluded(vehicle):
                continue

            bbox2 = get_vehicle_bbox_pixels(
                vehicle, self.camera, self.K, IMG_W, IMG_H
            )
            if bbox2 is None:
                continue
            x1, y1, x2, y2 = bbox2
            if (x2-x1) < MIN_BBOX_PX or (y2-y1) < MIN_BBOX_PX:
                continue

            if (self._frame_idx % CLASSIFY_INTERVAL == 0
                    or vid not in self._classify_cache):
                crop  = frame[y1:y2, x1:x2]
                vtype = self.classifier.classify(crop)
                self._classify_cache[vid] = vtype
            else:
                vtype = self._classify_cache[vid]

            v     = vehicle.get_velocity()
            speed = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
            results.append({"actor": vehicle, "type": vtype,
                             "bbox": bbox2, "speed": speed})
        return results

    # ─────────────────────────────────────────────────────────
    def _draw_camera_view(self, frame: np.ndarray, results: list) -> np.ndarray:
        for r in results:
            x1, y1, x2, y2 = r["bbox"]
            vtype = r["type"]
            speed = r["speed"]
            color = COLORS.get(vtype, COLORS["unknown"])

            loc  = r["actor"].get_location()
            dist = math.sqrt(
                (loc.x - self.cam_tf.location.x)**2 +
                (loc.y - self.cam_tf.location.y)**2 +
                (loc.z - self.cam_tf.location.z)**2
            )

            # BBox (항공뷰: 얇게, 2px)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

            # 라벨
            label = f"{ICON.get(vtype,'?')} {speed:.0f}km/h"
            (lw, lh), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            ly = max(y1 - 4, lh + 2)
            cv2.rectangle(frame, (x1, ly-lh-2), (x1+lw+4, ly+2), color, -1)
            cv2.putText(frame, label, (x1+2, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (0,0,0), 1, cv2.LINE_AA)

        # HUD 상단
        hud = frame.copy()
        cv2.rectangle(hud, (0,0), (IMG_W, 42), (0,0,0), -1)
        cv2.addWeighted(hud, 0.55, frame, 0.45, 0, frame)

        title = (f"V2I Aerial View  |  {self._map_name}"
                 f"  |  FPS {self._fps:.1f}  |  High={CAM_HEIGHT:.0f}m")
        cv2.putText(frame, title, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(frame, title, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0,200,255), 1, cv2.LINE_AA)

        # 우상단 카운트
        counts = {c: 0 for c in CLASSES}
        for r in results:
            if r["type"] in counts: counts[r["type"]] += 1
        px = IMG_W - 130
        cv2.rectangle(frame, (px-6,46), (IMG_W-4, 46+len(CLASSES)*22+24),
                      (0,0,0), -1)
        for i, cls in enumerate(CLASSES):
            cv2.putText(frame, f"{cls.upper()}: {counts[cls]}",
                        (px, 64+i*22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                        COLORS[cls], 2, cv2.LINE_AA)
        cv2.putText(frame, f"TOTAL: {len(results)}",
                    (px, 64+len(CLASSES)*22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220,220,220), 1)
        return frame

    # ─────────────────────────────────────────────────────────
    def _draw_vision_panel(self, frame: np.ndarray) -> np.ndarray:
        """좌하단에 MOG2 + CentroidTracker 디버그 패널"""
        dbg = self.vision_proc.get_debug_image()
        if dbg is None:
            return frame

        # dbg는 좌우 합성(2배 폭) → 480×135 썸네일
        panel_w, panel_h = 480, 135
        panel  = cv2.resize(dbg, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
        border = 2
        panel_b = cv2.copyMakeBorder(panel, border, border, border, border,
                                     cv2.BORDER_CONSTANT, value=(0,200,200))
        ph, pw = panel_b.shape[:2]

        h, w   = frame.shape[:2]
        margin = 10
        ys, xs = h - ph - margin, margin

        roi = frame[ys:ys+ph, xs:xs+pw].copy()
        cv2.addWeighted(roi, 0.2, np.zeros_like(roi), 0.8, 0, roi)
        frame[ys:ys+ph, xs:xs+pw] = roi
        frame[ys:ys+ph, xs:xs+pw] = panel_b

        cv2.putText(frame,
                    "[ VisionProcessor: MOG2 Mask | CentroidTracker ]",
                    (xs, ys-6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.40, (0,200,200), 1, cv2.LINE_AA)
        return frame

    # ─────────────────────────────────────────────────────────
    def _compose_display(self, cam_frame: np.ndarray,
                         radar: np.ndarray) -> np.ndarray:
        h, w = cam_frame.shape[:2]
        border = 2
        margin = 10

        # 3D 레이더 (우하단)
        framed = cv2.copyMakeBorder(
            radar, border, border, border, border,
            cv2.BORDER_CONSTANT, value=(0,160,255)
        )
        fh, fw = framed.shape[:2]
        ys = h - fh - margin
        xs = w - fw - margin

        roi = cam_frame[ys:ys+fh, xs:xs+fw].copy()
        cv2.addWeighted(roi, 0.18, np.zeros_like(roi), 0.82, 0, roi)
        cam_frame[ys:ys+fh, xs:xs+fw] = roi
        cam_frame[ys:ys+fh, xs:xs+fw] = framed

        cv2.putText(cam_frame, "[ 3D V2I RADAR (Open3D) ]",
                    (xs, ys-6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (0,160,255), 1, cv2.LINE_AA)

        # Vision 패널 (좌하단)
        cam_frame = self._draw_vision_panel(cam_frame)
        return cam_frame

    # ─────────────────────────────────────────────────────────
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        results = self._detect_and_classify(frame)

        if self._frame_idx % VISION_UPDATE_INTERVAL == 0:
            self.vision_proc.process_frame(frame)

        display = self._draw_camera_view(frame.copy(), results)
        radar   = self.radar.render(results)
        display = self._compose_display(display, radar)

        now          = time.time()
        self._fps    = 1.0 / max(now - self._t_last, 1e-6)
        self._t_last = now
        return display

    # ─────────────────────────────────────────────────────────
    def _respawn_npcs(self):
        print("[V2I] NPC 재소환 중...")
        for v in self.npc_list:
            try:
                if v.is_alive: v.destroy()
            except Exception:
                pass
        self.world.tick()
        self.npc_list = spawn_npc_vehicles(
            self.client, self.world, self.tm, NUM_NPC)
        self._classify_cache.clear()
        self._occlusion_cache.clear()

    # ─────────────────────────────────────────────────────────
    def run(self):
        cv2.namedWindow("V2I Aerial Monitor", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("V2I Aerial Monitor", IMG_W, IMG_H)
        print("[V2I] 관제 시작  q=종료  r=NPC재소환")

        while True:
            try:
                self.world.tick()
            except Exception as e:
                print(f"[V2I] tick 오류: {e}")
                break

            try:
                raw = self.img_queue.get_nowait()
            except queue.Empty:
                continue

            try:
                arr = np.frombuffer(raw.raw_data, np.uint8).reshape(
                    (raw.height, raw.width, 4)
                )[:, :, :3].copy()

                display = self.process_frame(arr)
                cv2.imshow("V2I Aerial Monitor", display)

            except Exception:
                traceback.print_exc()

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                self._respawn_npcs()

    # ─────────────────────────────────────────────────────────
    def cleanup(self):
        print("[V2I] 정리 중...")
        try:  self.camera.destroy()
        except Exception: pass

        batch = [carla.command.DestroyActor(v)
                 for v in self.npc_list if v.is_alive]
        if batch:
            self.client.apply_batch_sync(batch, True)

        try:
            for _ in range(3):
                self.world.tick()
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
        except Exception:
            pass

        try:  self.radar.destroy()
        except Exception: pass

        cv2.destroyAllWindows()
        print("[V2I] 종료 완료")


# =============================================
# 진입점
# =============================================
def main():
    system = V2IMonitorSystem()
    try:
        system.run()
    except KeyboardInterrupt:
        print("\n[V2I] 사용자 중단")
    except Exception:
        traceback.print_exc()
    finally:
        system.cleanup()


if __name__ == "__main__":
    main()
