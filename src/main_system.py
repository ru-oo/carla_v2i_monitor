"""
main_system.py
==============
담당: 전체 팀 (최종 통합 실행 파일)
브랜치: feature/cnn-model

역할:
    - CARLA V2I 스마트 교차로 3D 관제 시스템 메인 루프
    - 교차로 코너 CCTV (12m 높이 / 22m 오프셋 / pitch −25°)
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

from collections import deque, Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classifier import CLASSES, VehicleClassifier, LinearVehicleClassifier
from vision_processor import VisionProcessor


# =============================================
# 설정값
# =============================================
CARLA_HOST = "localhost"
CARLA_PORT = 2000
TM_PORT    = 9000

IMG_W      = 1280
IMG_H      = 720
CAM_FOV       = 110.0   # CCTV 화각
CAM_HEIGHT    = 12.0    # CCTV 폴 높이(시뮬레이션 값; 실제 코너 폴은 약 7m)
CAM_OFFSET    = 22.0    # 교차로 중심에서 수평 오프셋(대각선 분배: off=22/√2≈15.56m)
CAM_PITCH_DEG = -25.0   # 도로를 향해 내려다보는 각도
CAM_YAW_OFFSET = -50.0  # SW→NE 대각선(약 −45°)에 가산하는 yaw 보정

# 디스플레이 창 크기 (CCTV + 우측 패널)
DISP_W     = 1900       # 전체 창 너비 (CCTV 1280 + 우측 620)
DISP_H     = 720        # 전체 창 높이

# BEV 설정
BEV_PX     = 350        # BEV 이미지 크기 (정방형 px)
BEV_RANGE  = 50.0       # BEV 커버 범위 (m): 교차로 ±25m
NUM_NPC    = 40

MAP_PX     = 500        # Open3D 레이더 캔버스 크기

CLASSIFY_INTERVAL      = 3     # N프레임마다 재분류 (빠른 투표 반응)
MIN_BBOX_PX            = 8    # 항공뷰: 작은 BBox도 유효
RADAR_UPDATE_INTERVAL  = 6    # Open3D 갱신 주기 (프레임)
OCCLUSION_CACHE_FRAMES = 10
MAX_DETECTION_DIST     = 90.0
VISION_UPDATE_INTERVAL = 1    # VisionProcessor 매 프레임 실행 (속도 추정용)

# ── 라벨 안정화 ──────────────────────────────────────────────
VOTE_WINDOW    = 7     # 다수결 최근 N프레임
CONF_THRESHOLD = 0.55  # 신뢰도 미달 → 투표 불참
MIN_BBOX_CLS   = 20    # 이 픽셀 이하 BBox는 분류 생략

# ── 속도 추정 보정 ───────────────────────────────────────────
MIN_DISP_METER  = 0.30  # IPM 좌표 최소 변위 미만 → 정지 처리
SPEED_EMA_ALPHA = 0.15  # EMA 계수 (낮을수록 부드러움)
SPEED_CAP_KMH   = 80.0  # 속도 상한

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
    "car":     (0.24, 0.82, 0.00),
    "truck":   (1.00, 0.20, 0.20),
    "van":     (0.00, 0.71, 1.00),
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
# BEV (Bird's Eye View) 정사영 생성
# =============================================
def compute_bev(frame: np.ndarray,
                cam_inv: np.ndarray,
                K: np.ndarray,
                junc_x: float, junc_y: float, junc_z: float,
                bev_px: int   = BEV_PX,
                bev_range: float = BEV_RANGE) -> np.ndarray:
    """
    카메라 프레임 → BEV (North-up, 정사영법).

    각 BEV 픽셀에 대응하는 CARLA 지면 좌표를 계산하고,
    CARLA 카메라 투영으로 원본 이미지에서 색상을 샘플링.

    Args:
        frame    : BGR 카메라 이미지 (H×W×3)
        cam_inv  : CARLA 카메라 역변환 행렬 (4×4)
        K        : 카메라 내재 행렬 (3×3)
        junc_x/y/z: 교차로 중심 CARLA 월드 좌표
        bev_px   : 출력 BEV 크기 (정방형)
        bev_range: 커버할 범위 (m)  교차로 중심 ±bev_range/2

    Returns:
        BEV 이미지 (bev_px×bev_px×3, North-up)
    """
    h_img, w_img = frame.shape[:2]
    scale = bev_range / bev_px           # m/pixel

    # BEV 그리드: v=row(0=North), u=col(0=West)
    v_idx, u_idx = np.mgrid[0:bev_px, 0:bev_px]

    # BEV 픽셀 → CARLA 월드 좌표 (지면 평면 z=junc_z)
    wx = junc_x + (u_idx.ravel() - bev_px / 2.0) * scale   # East  = +X
    wy = junc_y + (v_idx.ravel() - bev_px / 2.0) * scale   # South = +Y (v=0→North)
    wz = np.full(bev_px * bev_px, junc_z)

    # CARLA 카메라 투영:  cam_inv @ [wx, wy, wz, 1]ᵀ
    ones      = np.ones(bev_px * bev_px)
    pts_world = np.stack([wx, wy, wz, ones], axis=1)   # (N,4)
    pts_cam   = (cam_inv @ pts_world.T).T               # (N,4)

    # CARLA 카메라 좌표: X=forward, Y=right, Z=up
    x_img = pts_cam[:, 1]
    y_img = -pts_cam[:, 2]
    z_dep = pts_cam[:, 0]

    valid = z_dep > 0.1
    safe_z = np.where(valid, z_dep, 1.0)
    px = (K[0, 0] * x_img / safe_z + K[0, 2]).astype(np.int32)
    py = (K[1, 1] * y_img / safe_z + K[1, 2]).astype(np.int32)

    in_bounds = valid & (px >= 0) & (px < w_img) & (py >= 0) & (py < h_img)

    bev_img = np.full((bev_px, bev_px, 3), 18, dtype=np.uint8)
    bev_img[v_idx.ravel()[in_bounds], u_idx.ravel()[in_bounds]] = \
        frame[py[in_bounds], px[in_bounds]]
    return bev_img


def draw_bev_overlay(bev_img: np.ndarray, road_hw: float) -> np.ndarray:
    """BEV 이미지에 도로 경계·거리 원호·방향 표시 오버레이."""
    h, w  = bev_img.shape[:2]
    out   = bev_img.copy()
    scale = BEV_RANGE / BEV_PX          # m/pixel
    cx, cy = w // 2, h // 2

    # 거리 원호
    for r_m in [10, 20, 25]:
        r_px = int(r_m / scale)
        cv2.circle(out, (cx, cy), r_px, (50, 72, 95), 1, cv2.LINE_AA)
        cv2.putText(out, f"{r_m}m", (cx + r_px + 2, cy - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (50, 72, 95), 1)

    # 도로 경계선 (측정 반폭)
    hw_px = max(1, int(road_hw / scale))
    col   = (80, 105, 130)
    cv2.line(out, (cx - hw_px, 0),  (cx - hw_px, h), col, 1)
    cv2.line(out, (cx + hw_px, 0),  (cx + hw_px, h), col, 1)
    cv2.line(out, (0, cy - hw_px),  (w, cy - hw_px), col, 1)
    cv2.line(out, (0, cy + hw_px),  (w, cy + hw_px), col, 1)

    # 교차로 중심 마커
    cv2.circle(out, (cx, cy), 5, (0, 200, 255), -1)

    # 방향 레이블
    cv2.putText(out, "N", (cx - 5, 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 200, 255), 1)
    cv2.putText(out, "S", (cx - 5, h - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 200, 255), 1)
    cv2.putText(out, "E", (w - 14, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 200, 255), 1)
    cv2.putText(out, "W", (2, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 200, 255), 1)
    return out


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
# CCTV 카메라 설치 (NW 코너 — 최적 위치 자동 선정)
# =============================================
def spawn_cctv_camera(world, junc_center):
    """
    교차로 NW 코너에서 SE(교차로 방향)를 바라보는 CCTV 설치.

    cctv_optimizer.py 자동 평가 결과 (Town10HD_Opt, NPC 40대):
        #1 NW corner  avg_visible=25.0  score=28.85  ← 채택
        #2 SW corner  avg_visible=20.5  score=24.29
        #8 South road avg_visible=15.7  score=19.63  (이전 위치)

    • 위치: 교차로 중심에서 NW 방향 (−X, −Y 각 CAM_OFFSET/√2 m)
    • 시선: 교차로 중심 방향(yaw은 CAM_YAW_OFFSET로 보정), pitch = −25°(CAM_PITCH_DEG)
    • FOV : 110° → North arm(좌상) + West arm(우하) + 교차로 전체 포착
    • 이유: 대각선 뷰로 2개 도로 암 + 교차로 전체가 시야에 들어와
            차량 인식 수가 South road 대비 약 60% 향상 (25.0 vs 15.7대)
    """
    bp_lib = world.get_blueprint_library()
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(IMG_W))
    cam_bp.set_attribute("image_size_y", str(IMG_H))
    cam_bp.set_attribute("fov", str(CAM_FOV))

    # SW 코너 위치 (좌측하단): −X(서쪽), +Y(남쪽)
    # 화면 기준: 좌하단 위치 → 우상단(NE) 방향을 바라봄
    off = CAM_OFFSET / math.sqrt(2)
    cx  = junc_center.x - off          # 서쪽(−X)
    cy  = junc_center.y + off          # 남쪽(+Y) ← NW에서 SW로 변경
    cz  = junc_center.z + CAM_HEIGHT

    # 수평 방향: SW → 교차로 중심(NE, yaw≈−45°) + 선택적 오프셋
    dx    = junc_center.x - cx         # +off (+X = East)
    dy    = junc_center.y - cy         # −off (−Y = North)
    yaw   = math.degrees(math.atan2(dy, dx)) + CAM_YAW_OFFSET  # ≈ −45° (NE)

    # 수직 방향: 살짝 위를 바라봄 → 교차로가 화면 하단에 위치
    # CAM_PITCH_DEG > 0: 교차로가 아래로 내려감 (화면 하단 90% 목표)
    pitch = CAM_PITCH_DEG

    cam_tf = carla.Transform(
        carla.Location(x=cx, y=cy, z=cz),
        carla.Rotation(pitch=pitch, yaw=yaw, roll=0.0),
    )

    img_q  = queue.Queue(maxsize=2)
    camera = world.spawn_actor(cam_bp, cam_tf)
    camera.listen(lambda img: img_q.put_nowait(img) if not img_q.full() else None)

    print(f"[V2I] CCTV 설치  위치=({cx:.1f}, {cy:.1f}, {cz:.1f}m)  "
          f"pitch={pitch:.1f}°  yaw={yaw:.1f}°  FOV={CAM_FOV}°")
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
        # light_on 기본값(True) 유지 — False 시 메쉬 색상이 검정으로 렌더됨

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
        MAP_OFFSET_X = -18.0
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
            hw = self.road_hw * 1.5
            ln = _3D_ROAD_LEN
            
            MAP_OFFSET_X = -18.0

            ROAD_COL   = [0.38, 0.43, 0.54]
            GROUND_COL = [0.04, 0.047, 0.071]
            ARC_COL    = [0.22, 0.28, 0.40]
            LANE_COL   = [0.50, 0.56, 0.68]
            CCTV_COL   = [0.0,  0.82, 0.54]
            CENTER_COL = [0.0,  0.78, 1.0 ]

            gnd = self._flat_rect(-65, -65, 65, 65, z=0.0, thick=0.05)
            gnd.translate([MAP_OFFSET_X, 0, 0])
            self._add(gnd, GROUND_COL)

            road_segs = [
                        (-hw, -2*ln, hw,  0),      # 북
                        (-hw,  0,  hw, ln),      # 남
                        ( 0, -3*hw, ln,  hw),    # 동 (화면 위쪽으로 폭 확장)
                        (-ln, -3*hw,  0, hw),    # 서 (화면 위쪽으로 폭 확장)
                        (-hw, -3*hw, hw, hw),    # 교차로 중앙 (위쪽 빈 공간 채움)
                    ]
            for i, (x1,y1,x2,y2) in enumerate(road_segs):
                r = self._flat_rect(x1, y1, x2, y2, z=0.01, thick=0.06)
                r.translate([MAP_OFFSET_X, 0, 0])
                self._add(r, ROAD_COL)

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
                ls.translate([MAP_OFFSET_X, 0, 0])
                self.vis.add_geometry(ls, reset_bounding_box=True)
                self._static_geoms.append(ls)

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
                ls.translate([MAP_OFFSET_X, 0, 0])
                self.vis.add_geometry(ls, reset_bounding_box=True)
                self._static_geoms.append(ls)

            cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=0.8, height=0.6)
            cyl.translate([MAP_OFFSET_X, 0, 0.3])
            self._add(cyl, CENTER_COL)

            cx, cy, cz = self._c(self.cam_loc.x, self.cam_loc.y, self.cam_loc.z)
            pole = o3d.geometry.LineSet()
            pole.points = o3d.utility.Vector3dVector([[cx,cy,0],[cx,cy,cz]])
            pole.lines  = o3d.utility.Vector2iVector([[0,1]])
            pole.colors = o3d.utility.Vector3dVector([CCTV_COL])
            self.vis.add_geometry(pole, reset_bounding_box=True)
            self._static_geoms.append(pole)

            ball = o3d.geometry.TriangleMesh.create_sphere(radius=1.2)
            ball.translate([cx, cy, cz])
            self._add(ball, CCTV_COL)

    def _apply_camera(self):
        ctr = self.vis.get_view_control()
        ctr.set_lookat([-15.6, -10.0, 0])
        ctr.set_up([0, 0, 1])
        ctr.set_front([0, 0.906, 0.424])
        ctr.set_zoom(0.30)

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
        # Z는 CARLA 월드좌표(차량 중심 높이)를 그대로 쓰면 도로면 위로 뜸
        # → XY만 사용하고 Z는 항상 도로 표면(0.05m)에 고정
        px, py, _ = self._c(loc.x, loc.y, loc.z)

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
            brightness     = 0.35 + 0.65 * z_frac   # 하단도 밝게 → 도로 위 차량 선명
            colors_arr[i]  = np.clip(base * brightness, 0, 1)
        box.vertex_colors = o3d.utility.Vector3dVector(colors_arr)
        box.compute_vertex_normals()

        # Z축 회전 (CARLA yaw = Open3D Z rotation 동방향)
        R = self._rot_z(yaw_rad)
        box.rotate(R, center=[0, 0, 0])
        # 도로 표면 z=0.01 (_flat_rect z=0.01이 TOP) → 차량 바닥을 도로에 정확히 붙임
        box.translate([px, py, 0.01])
        return box

    # ── V2I 신호선 (LineSet) ─────────────────────────────────
    def _make_v2i_lineset(self, results: list):
        pts   = [[0.0, 0.0, CAM_HEIGHT]]   # 교차로 중심 위 CCTV 높이의 V2I 노드(과거 z=-18은 지면 아래였음)
        lines = []
        cols  = []
        for r in results:
            loc = r["actor"].get_location()
            px, py, _ = self._c(loc.x, loc.y, loc.z)   # Z 무시
            if abs(px) > 53 or abs(py) > 53:
                continue
            idx = len(pts)
            pts.append([px, py, 0.5])   # 도로 표면에서 0.5m 위로 신호선
            lines.append([0, idx])
            base = _O3D_RGB.get(r.get("cnn_type", r.get("type", "unknown")), _O3D_RGB["unknown"])
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
            box = self._make_vehicle_box(r["actor"], r.get("cnn_type", r.get("type", "unknown")))
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
        bgr       = cv2.flip(bgr, 1)           # 좌우 미러 보정: Open3D에서 East(+X)가 왼쪽으로 나오는 현상 수정

        if bgr.shape[:2] != (MAP_PX, MAP_PX):
            bgr = cv2.resize(bgr, (MAP_PX, MAP_PX))

        # ── cv2 오버레이 (범례·거리 라벨) ─────────────────────
        counts = {c: 0 for c in CLASSES}
        for r in results:
            if r.get("cnn_type", r.get("type", "unknown")) in counts:
                counts[r.get("cnn_type", r.get("type", "unknown"))] += 1
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

        # 카메라 역행렬 사전 계산 (카메라는 정적으로 위치 고정)
        self._cam_inv_mat = np.array(
            self.camera.get_transform().get_inverse_matrix()
        )

        # ── CNN 분류기 + Linear 비교 분류기 ─────────────────────
        self.classifier     = VehicleClassifier()
        self.lin_classifier = None
        try:
            self.lin_classifier = LinearVehicleClassifier()
        except FileNotFoundError:
            print("[V2I] Linear 모델 없음 — CNN만 사용 (python src/classifier.py 로 학습)")

        # ── 라벨 투표 캐시 (플리커링 방지) ──────────────────────
        # {vehicle_id: deque([label, label, ...], maxlen=VOTE_WINDOW)}
        self._vote_cnn: dict[int, deque] = {}   # CNN 투표
        self._vote_lin: dict[int, deque] = {}   # Linear 투표
        self._stable_cnn: dict[int, str] = {}   # CNN 확정 라벨
        self._stable_lin: dict[int, str] = {}   # Linear 확정 라벨

        self._occlusion_cache: dict[int, tuple[bool,int]] = {}
        self._frame_idx = 0
        self._vision_speed_cache: dict[int, float] = {}   # {carla_vid: vision_kmh}
        self._vision_speed_frame:  dict[int, int]  = {}   # {carla_vid: last_matched_frame}

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
    def _vote_update(self, vid: int, cnn_lbl: str, cnn_conf: float,
                     lin_lbl: str, lin_conf: float):
        """투표 캐시 업데이트 → stable 라벨 갱신"""
        # CNN 투표
        if vid not in self._vote_cnn:
            self._vote_cnn[vid] = deque(maxlen=VOTE_WINDOW)
        if cnn_lbl != "uncertain" and cnn_conf >= CONF_THRESHOLD:
            self._vote_cnn[vid].append(cnn_lbl)

        # Linear 투표
        if vid not in self._vote_lin:
            self._vote_lin[vid] = deque(maxlen=VOTE_WINDOW)
        if lin_lbl != "uncertain" and lin_conf >= CONF_THRESHOLD:
            self._vote_lin[vid].append(lin_lbl)

        def _majority(q, prev):
            if not q:
                return prev or "unknown"
            cnt = Counter(q)
            top_lbl, top_cnt = cnt.most_common(1)[0]
            # 40% 이상 지지 시 확정
            return top_lbl if top_cnt / len(q) >= 0.40 else (prev or "unknown")

        self._stable_cnn[vid] = _majority(
            self._vote_cnn[vid], self._stable_cnn.get(vid))
        self._stable_lin[vid] = _majority(
            self._vote_lin[vid], self._stable_lin.get(vid))

    def _vote_cleanup(self, active_ids: set):
        """사라진 차량의 투표 캐시 정리"""
        for d in (self._vote_cnn, self._vote_lin,
                  self._stable_cnn, self._stable_lin):
            for old_id in list(d):
                if old_id not in active_ids:
                    del d[old_id]

    # ─────────────────────────────────────────────────────────
    def _detect_and_classify(self, frame: np.ndarray) -> list:
        """
        CARLA actor 목록에서 가시 차량 검출 + 분류.

        결과 dict 키:
            actor       - CARLA Vehicle actor
            bbox        - (x1,y1,x2,y2) 픽셀 BBox
            gt_speed    - CARLA get_velocity() 기반 정답 속도 (km/h)
            cnn_type    - CNN 투표 확정 라벨
            lin_type    - Linear 투표 확정 라벨 (Linear 없으면 CNN과 동일)
        """
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

            # ── 분류 (CLASSIFY_INTERVAL마다 또는 첫 등장) ─────────
            needs_cls = (
                self._frame_idx % CLASSIFY_INTERVAL == 0
                or vid not in self._stable_cnn
            )
            if needs_cls:
                bw, bh = x2-x1, y2-y1
                if bw >= MIN_BBOX_CLS and bh >= MIN_BBOX_CLS:
                    crop = frame[y1:y2, x1:x2]
                    cnn_lbl, cnn_conf = self.classifier.classify_with_conf(crop)
                    if self.lin_classifier is not None:
                        lin_lbl, lin_conf = self.lin_classifier.classify_with_conf(crop)
                    else:
                        lin_lbl, lin_conf = cnn_lbl, cnn_conf
                else:
                    # BBox 너무 작으면 기본값으로 투표
                    cnn_lbl, cnn_conf = "car", 0.6
                    lin_lbl, lin_conf = "car", 0.6

                self._vote_update(vid, cnn_lbl, cnn_conf, lin_lbl, lin_conf)

            cnn_type = self._stable_cnn.get(vid, "unknown")
            lin_type = self._stable_lin.get(vid, "unknown")

            # ── GT 속도 (CARLA 시뮬레이터 정답값) ─────────────────
            v        = vehicle.get_velocity()
            gt_speed = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)

            results.append({
                "actor":    vehicle,
                "bbox":     bbox2,
                "gt_speed": gt_speed,
                "cnn_type": cnn_type,
                "lin_type": lin_type,
            })

        # 사라진 차량 캐시 정리
        active_ids = {r["actor"].id for r in results}
        self._vote_cleanup(active_ids)

        return results

    # ─────────────────────────────────────────────────────────
    def _draw_camera_view(self, frame: np.ndarray, results: list) -> np.ndarray:
        """
        BBox + 라벨 오버레이.

        라벨 구조 (2줄):
            윗줄: CNN 결과  [C bus  32km/h]  — CNN 색상
            아랫줄: Linear  [L car        ]  — Linear 색상, 불일치 시 빨간 배경

        정지 차량 노이즈 억제:
            - gt_speed < 2.0 km/h → "0 km/h" 로 표시 (스냅)
        """
        counts_cnn = {c: 0 for c in CLASSES}
        counts_lin = {c: 0 for c in CLASSES}
        mismatch   = 0

        for r in results:
            x1, y1, x2, y2 = r["bbox"]
            cnn_type = r["cnn_type"]
            lin_type = r["lin_type"]
            gt_speed = r["gt_speed"]

            # ── 정지 노이즈 억제: 5km/h 미만은 0으로 스냅 (Vision과 통일) ───────────
            display_speed = 0.0 if gt_speed < 5.0 else gt_speed

            cnn_col = COLORS.get(cnn_type, COLORS["unknown"])
            lin_col = COLORS.get(lin_type, COLORS["unknown"])
            is_diff = (cnn_type != lin_type
                       and cnn_type not in ("unknown",)
                       and lin_type not in ("unknown",))
            if is_diff:
                mismatch += 1

            if cnn_type in counts_cnn: counts_cnn[cnn_type] += 1
            if lin_type in counts_lin: counts_lin[lin_type] += 1

            # ── BBox: 불일치 시 노란 테두리 강조 ─────────────────────
            bbox_col = (0, 220, 220) if is_diff else cnn_col
            cv2.rectangle(frame, (x1,y1), (x2,y2), bbox_col, 2)

            fs = 0.36; th = 1
            vision_speed = r.get("vision_speed")  # None = IoU 미매핑

            # ── 줄1: CNN 차종 + GT 속도 (CARLA 정답) ─────────────────
            cnn_txt = f"C:{ICON.get(cnn_type,'?')}{cnn_type}  GT:{display_speed:.0f}"
            (tw, tlh), _ = cv2.getTextSize(cnn_txt, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
            gap = tlh + 5
            ty1 = max(y1 - gap*3 - 4, tlh + 4)
            ty2 = ty1 + gap
            ty3 = ty2 + gap

            cv2.rectangle(frame, (x1, ty1-tlh-2), (x1+tw+4, ty1+2), (15,15,15), -1)
            cv2.rectangle(frame, (x1, ty1-tlh-2), (x1+tw+4, ty1+2), cnn_col, 1)
            cv2.putText(frame, cnn_txt, (x1+2, ty1),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, cnn_col, th, cv2.LINE_AA)

            # ── 줄2: Vision 추정 속도 + 오차 ─────────────────────────
            if vision_speed is not None:
                err = abs(display_speed - vision_speed)
                # 오차 색상: ≤5 초록 / ≤15 노랑 / 초과 빨강
                spd_col = (0,210,80) if err<=5 else (0,200,255) if err<=15 else (50,50,255)
                vis_txt = f"V:{vision_speed:.0f}  Err:{err:.0f}km/h"
            else:
                spd_col = (100, 100, 100)
                vis_txt = "V:--"
            (tw_v, _), _ = cv2.getTextSize(vis_txt, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
            cv2.rectangle(frame, (x1, ty2-tlh-2), (x1+tw_v+4, ty2+2), (15,15,15), -1)
            cv2.rectangle(frame, (x1, ty2-tlh-2), (x1+tw_v+4, ty2+2), spd_col, 1)
            cv2.putText(frame, vis_txt, (x1+2, ty2),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, spd_col, th, cv2.LINE_AA)

            # ── 줄3: Linear 차종 (불일치 시 붉은 배경) ───────────────
            lin_txt = f"L:{ICON.get(lin_type,'?')}{lin_type}"
            (tw2, _), _ = cv2.getTextSize(lin_txt, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
            bg_col = (50, 15, 15) if is_diff else (15, 15, 15)
            cv2.rectangle(frame, (x1, ty3-tlh-2), (x1+tw2+4, ty3+2), bg_col, -1)
            cv2.rectangle(frame, (x1, ty3-tlh-2), (x1+tw2+4, ty3+2), lin_col, 1)
            cv2.putText(frame, lin_txt, (x1+2, ty3),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, lin_col, th, cv2.LINE_AA)

        # ── 상단 HUD ─────────────────────────────────────────────────
        hud = frame.copy()
        cv2.rectangle(hud, (0,0), (IMG_W, 44), (0,0,0), -1)
        cv2.addWeighted(hud, 0.55, frame, 0.45, 0, frame)

        title = (f"V2I CCTV Monitor  |  {self._map_name}"
                 f"  |  FPS {self._fps:.1f}"
                 f"  |  H={CAM_HEIGHT:.0f}m / off={CAM_OFFSET:.0f}m")
        cv2.putText(frame, title, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(frame, title, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (0,200,255), 1, cv2.LINE_AA)

        # ── 우상단: CNN / Linear 카운트 나란히 ───────────────────────
        px = IMG_W - 295
        box_h = len(CLASSES)*24 + 68
        cv2.rectangle(frame, (px-6,46), (IMG_W-4, 46+box_h), (0,0,0), -1)

        # 열 헤더
        cv2.putText(frame, "CNN",    (px+10, 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100,200,255), 1)
        cv2.putText(frame, "LINEAR", (px+120, 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,120,100), 1)
        cv2.line(frame, (px-2,68), (IMG_W-4,68), (60,60,80), 1)

        for i, cls in enumerate(CLASSES):
            y_t = 82 + i*24
            col = COLORS[cls]
            cv2.putText(frame, f"{cls.upper()}: {counts_cnn[cls]}",
                        (px, y_t),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, col, 1)
            cv2.putText(frame, str(counts_lin.get(cls, 0)),
                        (px+155, y_t),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, col, 1)

        sep_y = 82 + len(CLASSES)*24 + 2
        cv2.line(frame, (px-2, sep_y), (IMG_W-4, sep_y), (60,60,80), 1)
        cv2.putText(frame,
                    f"TOTAL: {len(results)}   불일치: {mismatch}",
                    (px, sep_y+16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,200), 1)

        # ── 라벨 범례 (좌하단 작게) ──────────────────────────────────
        cv2.putText(frame,
                    "C=CNN(blue)  L=Linear(red)  yellow=mismatch",
                    (8, IMG_H-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (120,140,160), 1)

        return frame

    # ─────────────────────────────────────────────────────────
    def _compose_display(self,
                         cam_frame: np.ndarray,
                         radar:     np.ndarray,
                         bev:       np.ndarray,
                         dbg:       np.ndarray) -> np.ndarray:
        """
        1900×720 캔버스에 4개 패널을 분리 배치 (겹침 없음):

          ┌───────────────────────┬────────────────────┐
          │ CCTV View  1280×720   │ 3D V2I RADAR       │
          │                       │      618×362        │
          │                       ├────────────────────┤
          │                       │ BEV  618×178        │
          │                       ├────────────────────┤
          │                       │ MOG2+Tracker 618×178│
          └───────────────────────┴────────────────────┘
        """
        canvas = np.zeros((DISP_H, DISP_W, 3), dtype=np.uint8)

        # ── 좌측: CCTV 뷰 (1280×720) ──────────────────────────
        canvas[:DISP_H, :IMG_W] = cam_frame

        # 수직 구분선
        canvas[:, IMG_W:IMG_W+2] = (55, 60, 72)
        RX = IMG_W + 2                    # 우측 패널 시작 X (= 1282)
        RW = DISP_W - RX                  # 우측 패널 너비 (≈ 618 px)

        SEP = (55, 60, 72)  # 구분선 색

        # ── 우측 상단: 3D V2I Radar ───────────────────────────
        RAD_H = 362
        rad_s = cv2.resize(radar, (RW, RAD_H))
        canvas[:RAD_H, RX:RX+RW] = rad_s
        cv2.putText(canvas, "[ 3D V2I RADAR ]",
                    (RX+8, RAD_H-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 160, 255), 1, cv2.LINE_AA)
        canvas[RAD_H:RAD_H+2, RX:] = SEP

        # ── 우측 중간: BEV (Ground Projection) ───────────────
        BEV_TOP = RAD_H + 2
        BEV_H   = 178
        bev_sq  = BEV_H - 4               # 정방형 BEV 크기
        bev_s   = cv2.resize(bev, (bev_sq, bev_sq))
        bev_panel = np.zeros((BEV_H, RW, 3), dtype=np.uint8)
        bev_panel[2:2+bev_sq, 2:2+bev_sq] = bev_s
        # BEV 오른쪽 여백에 레이블
        cv2.putText(bev_panel, "[ BEV / Ground Projection ]",
                    (bev_sq+8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 200, 255), 1, cv2.LINE_AA)
        cv2.putText(bev_panel, f"North-up  {BEV_RANGE:.0f}x{BEV_RANGE:.0f}m",
                    (bev_sq+8, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (120, 140, 160), 1, cv2.LINE_AA)
        canvas[BEV_TOP:BEV_TOP+BEV_H, RX:RX+RW] = bev_panel
        canvas[BEV_TOP+BEV_H:BEV_TOP+BEV_H+2, RX:] = SEP

        # ── 우측 하단: MOG2 Binary + CentroidTracker ─────────
        DBG_TOP = BEV_TOP + BEV_H + 2
        DBG_H   = DISP_H - DBG_TOP       # ≈ 178 px
        if dbg is not None and DBG_H > 10:
            dbg_s = cv2.resize(dbg, (RW, DBG_H))
            canvas[DBG_TOP:DBG_TOP+DBG_H, RX:RX+RW] = dbg_s

        return canvas

    # ─────────────────────────────────────────────────────────
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        results = self._detect_and_classify(frame)

        # VisionProcessor → TrackedObject BBox를 CARLA BBox에 IoU 매핑
        # → results 각 항목에 'vision_speed' 키 추가
        tracked = self.vision_proc.process_frame(frame)

        # TrackedObject (x,y,w,h) → (x1,y1,x2,y2) 변환
        vis_boxes = []
        for t in tracked:
            vx, vy, vw, vh = t.bbox
            vis_boxes.append((vx, vy, vx+vw, vy+vh, t.speed_kmh))

        matched_r = set(); matched_v = set()
        if results and vis_boxes:
            def _iou(a, b):
                ix1,iy1 = max(a[0],b[0]), max(a[1],b[1])
                ix2,iy2 = min(a[2],b[2]), min(a[3],b[3])
                inter = max(0,ix2-ix1)*max(0,iy2-iy1)
                if inter==0: return 0.0
                return inter/max((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter,1e-6)
            iou_mat = [[_iou(r["bbox"], vb[:4]) for vb in vis_boxes] for r in results]
            flat = sorted(range(len(results)*len(vis_boxes)),
                          key=lambda k: -iou_mat[k//len(vis_boxes)][k%len(vis_boxes)])
            for k in flat:
                ci, vi = k//len(vis_boxes), k%len(vis_boxes)
                if iou_mat[ci][vi] < 0.20: break
                if ci in matched_r or vi in matched_v: continue
                matched_r.add(ci); matched_v.add(vi)
                self._vision_speed_cache[results[ci]["actor"].id] = vis_boxes[vi][4]
                self._vision_speed_frame[results[ci]["actor"].id] = self._frame_idx

        # [v4] 매칭 안 된 차량의 캐시 속도: 3프레임 이상 미매칭이면 None 처리
        # (정지 차량이 MOG2에서 사라져도 이전 속도가 display되는 문제 방지)
        VISION_STALE_FRAMES = 3
        active = {r["actor"].id for r in results}
        for r in results:
            vid  = r["actor"].id
            last = self._vision_speed_frame.get(vid, -999)
            if self._frame_idx - last <= VISION_STALE_FRAMES:
                r["vision_speed"] = self._vision_speed_cache.get(vid)
            else:
                r["vision_speed"] = None   # 오래된 캐시는 사용 안 함
        for old in list(self._vision_speed_cache):
            if old not in active:
                del self._vision_speed_cache[old]
                self._vision_speed_frame.pop(old, None)

        display = self._draw_camera_view(frame.copy(), results)
        # V2IMapRenderer는 cnn_type 키로 차종 색상 결정
        radar   = self.radar.render(results)

        # BEV 정사영 (사전 계산된 cam_inv_mat 재사용 → 빠름)
        bev = compute_bev(
            frame,
            self._cam_inv_mat,
            self.K,
            self.junction_center.x,
            self.junction_center.y,
            self.junction_center.z,
        )
        bev = draw_bev_overlay(bev, self.radar.road_hw)

        dbg     = self.vision_proc.get_debug_image()
        output  = self._compose_display(display, radar, bev, dbg)

        now          = time.time()
        self._fps    = 1.0 / max(now - self._t_last, 1e-6)
        self._t_last = now
        return output

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
        # 투표 캐시 + 확정 라벨 모두 초기화
        self._vote_cnn.clear()
        self._vote_lin.clear()
        self._stable_cnn.clear()
        self._stable_lin.clear()
        self._occlusion_cache.clear()

    # ─────────────────────────────────────────────────────────
    def run(self):
        cv2.namedWindow("V2I Monitor", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("V2I Monitor", DISP_W, DISP_H)
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
                cv2.imshow("V2I Monitor", display)

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