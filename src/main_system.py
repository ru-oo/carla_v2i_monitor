"""
main_system.py
==============
담당: 전체 팀 (최종 통합 실행 파일)
브랜치: feature/cnn-model

역할:
    - CARLA V2I 스마트 교차로 3D 관제 시스템 메인 루프
    - CARLA actor list 직접 투영 → 4클래스 CNN 분류 → 3D 레이더 맵 렌더링
    - data_collector.py에서 검증된 교차로 탐색 / 카메라 설치 로직 재사용

실행 방법:
    python src/main_system.py

조작:
    q — 종료
    r — NPC 차량 재소환
"""

import math
import os
import queue
import random
import sys
import time

import carla
import cv2
import numpy as np

# classifier.py는 같은 src/ 폴더에 있으므로 경로 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classifier import CLASSES, VehicleClassifier

# =============================================
# 설정값
# =============================================
CARLA_HOST = "localhost"
CARLA_PORT = 2000
TM_PORT    = 9000       # main_system 전용 TM 포트 (data_collector와 충돌 방지)

IMG_W      = 1280
IMG_H      = 720
CAM_FOV    = 90.0
CAM_HEIGHT = 8.0        # CCTV 신호등 폴 높이 (m) — 현실적인 교차로 카메라
CAM_OFFSET = 18.0       # 교차로 중심 대비 수평 거리 (m) — 코너 폴 위치
NUM_NPC    = 40         # NPC 차량 수

MAP_PX     = 500        # 3D 레이더 맵 캔버스 크기 (px)

CLASSIFY_INTERVAL = 5   # N프레임마다 CNN 재분류 (CPU 부하 완화)
MIN_BBOX_PX       = 12  # 유효 BBox 최소 변 길이 (px)

# ─── 3D 레이더 가상 카메라 (Tesla 계기판 시점) ───
# 교차로 남쪽 55 m + 높이 35 m → 북쪽(교차로 방향) 내려다봄
# pitch ≈ -32°  → 도로가 원근으로 수렴하는 자동차 계기판 느낌
_3D_EYE_Y_OFFSET = 55.0    # 교차로 남쪽으로 (CARLA +Y = 남)
_3D_EYE_Z        = 35.0    # 고도 (m)
_3D_VFOV_DEG     = 55.0    # 수직 FOV
_3D_ROAD_HW      = 6.5     # 도로 반폭 (m) — 교차로 4방향 도로 그리기용
_3D_ROAD_LEN     = 55.0    # 각 도로 암 길이 (m)

# ─── 차종별 시각화 색상 (BGR) ───
COLORS = {
    "bus":     (0,   140, 255),   # 주황
    "car":     (0,   210,  60),   # 초록
    "truck":   (50,   50, 255),   # 빨강
    "van":     (255, 180,   0),   # 하늘
    "unknown": (150, 150, 150),   # 회색
}
ICON = {"bus": "B", "car": "C", "truck": "T", "van": "V", "unknown": "?"}


# =============================================
# 카메라 투영 유틸리티  (data_collector.py와 동일 로직)
# =============================================
def build_projection_matrix(w: int, h: int, fov: float) -> np.ndarray:
    """CARLA 카메라 내부 행렬(Intrinsic Matrix)"""
    focal = w / (2.0 * math.tan(math.radians(fov) / 2.0))
    return np.array([
        [focal, 0,     w / 2.0],
        [0,     focal, h / 2.0],
        [0,     0,     1.0    ],
    ])


def world_to_pixel(world_loc, camera_actor, K: np.ndarray):
    """
    3D 월드 좌표 → 2D 카메라 픽셀 좌표

    Returns:
        (u, v) 또는 카메라 뒤쪽이면 None
    """
    cam_inv = np.array(camera_actor.get_transform().get_inverse_matrix())
    pt      = cam_inv @ np.array([world_loc.x, world_loc.y, world_loc.z, 1.0])

    # UE4 → 이미지 좌표계:  forward=x, right=y, up=z  →  u=y, v=-z
    x_img = pt[1]
    y_img = -pt[2]
    z_img = pt[0]   # depth (양수 = 카메라 앞)

    if z_img <= 0:
        return None

    u = int(K[0, 0] * (x_img / z_img) + K[0, 2])
    v = int(K[1, 1] * (y_img / z_img) + K[1, 2])
    return u, v


def get_vehicle_bbox_pixels(vehicle, camera_actor, K, img_w: int, img_h: int):
    """
    차량 3D BBox 8꼭짓점 → 2D 픽셀 BBox (x1, y1, x2, y2)

    Returns:
        (x1, y1, x2, y2)  클리핑 후 유효 범위
        None  if  화면 밖 또는 카메라 뒤
    """
    verts = vehicle.bounding_box.get_world_vertices(vehicle.get_transform())
    us, vs = [], []
    for vert in verts:
        px = world_to_pixel(vert, camera_actor, K)
        if px is not None:
            us.append(px[0])
            vs.append(px[1])

    if len(us) < 2:
        return None

    x1, y1 = min(us), min(vs)
    x2, y2 = max(us), max(vs)

    # 화면 밖이면 제거
    if x2 < 0 or y2 < 0 or x1 >= img_w or y1 >= img_h:
        return None

    # 클리핑
    return (max(x1, 0), max(y1, 0), min(x2, img_w - 1), min(y2, img_h - 1))


# =============================================
# 교차로 / CCTV 유틸리티  (data_collector.py와 동일 로직)
# =============================================
def find_intersection(world) -> carla.Location:
    """도심 대표 교차로 위치 탐색 — junction waypoint 평균 중심 기준"""
    carla_map = world.get_map()
    all_wps   = carla_map.generate_waypoints(5.0)
    junc_wps  = [wp for wp in all_wps if wp.is_junction]

    if not junc_wps:
        sps = carla_map.get_spawn_points()
        cx  = sum(s.location.x for s in sps) / len(sps)
        cy  = sum(s.location.y for s in sps) / len(sps)
        print("[V2I] junction 없음 → spawn point 평균 사용")
        return carla.Location(x=cx, y=cy, z=0.0)

    cx = sum(wp.transform.location.x for wp in junc_wps) / len(junc_wps)
    cy = sum(wp.transform.location.y for wp in junc_wps) / len(junc_wps)
    best = min(junc_wps,
               key=lambda wp: (wp.transform.location.x - cx) ** 2
                            + (wp.transform.location.y - cy) ** 2)
    loc = best.transform.location
    print(f"[V2I] 교차로 탐색: ({loc.x:.1f}, {loc.y:.1f})")
    return carla.Location(x=loc.x, y=loc.y, z=loc.z)


def find_cctv_mount_position(world, junc_center: carla.Location) -> carla.Location:
    """
    교차로 SW 진입로 도로 위 CCTV 마운트 좌표
    (건물 내부 스폰 방지 — junction API로 실제 도로 위 waypoint 획득)
    """
    carla_map = world.get_map()
    junction  = None
    for ox, oy in [(0,0),(3,0),(-3,0),(0,3),(0,-3),(5,5),(-5,5),(5,-5),(-5,-5)]:
        test = carla.Location(x=junc_center.x+ox, y=junc_center.y+oy, z=junc_center.z)
        wp   = carla_map.get_waypoint(test, project_to_road=True)
        if wp and wp.is_junction:
            junction = wp.get_junction()
            break

    if junction is None:
        sw   = carla.Location(x=junc_center.x - 15, y=junc_center.y + 15, z=junc_center.z)
        snap = carla_map.get_waypoint(sw, project_to_road=True)
        return snap.transform.location if snap else junc_center

    best_wp, best_score = None, -float("inf")
    for entry_wp, _ in junction.get_waypoints(carla.LaneType.Driving):
        loc   = entry_wp.transform.location
        score = (junc_center.x - loc.x) + (loc.y - junc_center.y)
        if score > best_score:
            best_score = score
            best_wp    = entry_wp

    if best_wp is None:
        return junc_center

    # 교차로 경계에서 15 m 후퇴 → 카메라 pitch 각도 확보
    prev_wps = best_wp.previous(15.0)
    return prev_wps[0].transform.location if prev_wps else best_wp.transform.location


def spawn_cctv_camera(world, junc_center: carla.Location):
    """
    현실적인 교차로 CCTV 카메라 설치

    배치 방식:
        - 교차로 중심 SW 방향 CAM_OFFSET(18 m) 거리, CAM_HEIGHT(8 m) 신호등 폴 높이
        - pitch ≈ -24° (traffic camera 전형적 각도)
        - 교차로 진입 차량을 정면/측면으로 포착하는 실제 CCTV 시야

    Returns:
        (camera_actor, img_queue, cam_transform)
    """
    bp_lib = world.get_blueprint_library()
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(IMG_W))
    cam_bp.set_attribute("image_size_y", str(IMG_H))
    cam_bp.set_attribute("fov", str(CAM_FOV))

    # SW 방향 = x - offset*cos45, y + offset*sin45
    off = CAM_OFFSET * 0.7071
    cx  = junc_center.x - off
    cy  = junc_center.y + off
    cz  = junc_center.z + CAM_HEIGHT     # 8 m 신호등 폴

    dx    = junc_center.x - cx
    dy    = junc_center.y - cy
    dz    = junc_center.z - cz           # 음수 (-8 m)
    horiz = math.sqrt(dx * dx + dy * dy) # ≈ 18 m

    yaw   = math.degrees(math.atan2(dy, dx))
    pitch = math.degrees(math.atan2(dz, horiz))   # ≈ -24°

    cam_transform = carla.Transform(
        carla.Location(x=cx, y=cy, z=cz),
        carla.Rotation(pitch=pitch, yaw=yaw, roll=0.0),
    )

    img_q  = queue.Queue(maxsize=2)
    camera = world.spawn_actor(cam_bp, cam_transform)
    camera.listen(lambda img: img_q.put_nowait(img) if not img_q.full() else None)

    print(f"[V2I] CCTV 설치 완료 (교차로 CCTV 폴)\n"
          f"  위치: ({cx:.1f}, {cy:.1f}, {cz:.1f}m)\n"
          f"  pitch={pitch:.1f}°  yaw={yaw:.1f}°  교차로까지 {horiz:.1f}m")
    return camera, img_q, cam_transform


def spawn_npc_vehicles(client, world, tm, num: int = NUM_NPC) -> list:
    """NPC 차량 소환 (4륜+ 한정, 자율주행 활성화)"""
    bp_lib  = world.get_blueprint_library()
    all_bps = [bp for bp in bp_lib.filter("vehicle.*")
               if int(bp.get_attribute("number_of_wheels")) >= 4]
    spawn_pts = world.get_map().get_spawn_points()
    random.shuffle(spawn_pts)

    batch = []
    for sp in spawn_pts[:num]:
        bp = random.choice(all_bps)
        batch.append(
            carla.command.SpawnActor(bp, sp).then(
                carla.command.SetAutopilot(carla.command.FutureActor, True, tm.get_port())
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

    print(f"[V2I] NPC 소환 완료: {len(vehicles)}대")
    return vehicles


# =============================================
# 3D V2I 레이더 맵 렌더러
# =============================================
class V2IMapRenderer:
    """
    Tesla FSD 계기판 스타일 3D V2I 레이더 뷰 (MAP_PX × MAP_PX)

    ┌─ 원근 투영 설계 ────────────────────────────────────┐
    │  가상 카메라: 교차로 남쪽 55 m + 높이 35 m           │
    │  시선 방향: 북쪽(교차로 방향)으로 약 -32° 내려다봄   │
    │  → 도로가 화면 상단 소실점으로 수렴하는 원근 효과    │
    │  → 차량이 접근할수록 크게, 멀수록 작게 표시          │
    └──────────────────────────────────────────────────────┘

    시각 요소:
        - 어두운 Tesla 배경 (near-black, 청색 계열)
        - 원근 도로 (4방향 암 + 교차로 중앙 박스)
        - 지면 거리 원호 (10 / 20 / 30 / 40 m)
        - 3D 차량 박스 (CARLA bounding box 원근 투영, 차종별 색상)
        - 진행 방향 화살표
        - V2I 신호선 (교차로 중심 → 각 차량, 점선)
        - CCTV 폴 마커
        - 차종별 카운트 범례
    """

    # ── Tesla 색상 팔레트 ──────────────────────
    _BG      = (10,  12,  18)    # 배경 (매우 어두운 남색)
    _ROAD    = (38,  42,  52)    # 도로 아스팔트
    _LANE    = (70,  80,  95)    # 차선
    _GRID    = (22,  26,  34)    # 거리 원호
    _CENTER  = (0,  200, 255)    # 교차로 중심 강조색
    _CCTV    = (0,  210, 140)    # CCTV 마커

    def __init__(self, junction_center: carla.Location,
                 cam_location: carla.Location):
        self.junc    = junction_center
        self.cam_loc = cam_location

        # 가상 카메라 행렬 구성
        jx, jy, jz  = junction_center.x, junction_center.y, junction_center.z
        eye = np.array([jx,
                        jy + _3D_EYE_Y_OFFSET,   # 55 m 남쪽
                        jz + _3D_EYE_Z],          # 35 m 고도
                       dtype=float)
        target = np.array([jx, jy, jz], dtype=float)

        fwd  = target - eye
        fwd /= np.linalg.norm(fwd)

        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(world_up, fwd)         # OpenGL: up × fwd = right
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        right /= np.linalg.norm(right)

        cam_up = np.cross(fwd, right)
        cam_up /= np.linalg.norm(cam_up)

        self._eye    = eye
        self._right  = right
        self._cam_up = cam_up
        self._fwd    = fwd
        self._focal  = (MAP_PX / 2) / math.tan(math.radians(_3D_VFOV_DEG) / 2)
        self._cx     = MAP_PX // 2
        self._cy     = MAP_PX // 2

    # ── 핵심 투영 함수 ─────────────────────────
    def _proj(self, wx: float, wy: float, wz: float = 0.0):
        """3D 월드 좌표 → 캔버스 2D 픽셀. 카메라 뒤면 None."""
        rel = np.array([wx - self._eye[0],
                        wy - self._eye[1],
                        wz - self._eye[2]])
        px = float(np.dot(self._right,  rel))
        py = float(np.dot(self._cam_up, rel))
        pz = float(np.dot(self._fwd,    rel))
        if pz <= 0.1:
            return None
        u = int(self._cx + self._focal * px / pz)
        v = int(self._cy - self._focal * py / pz)   # 화면 Y 반전
        return u, v

    def _in_canvas(self, pt) -> bool:
        return pt is not None and 0 <= pt[0] < MAP_PX and 0 <= pt[1] < MAP_PX

    # ── 도로 ───────────────────────────────────
    def _draw_road(self, img: np.ndarray):
        jx, jy = self.junc.x, self.junc.y
        hw  = _3D_ROAD_HW
        ln  = _3D_ROAD_LEN

        # 4방향 도로 암: (좌측좌표, 우측좌표) 리스트
        arms = [
            [(jx-hw, jy),    (jx-hw, jy-ln), (jx+hw, jy-ln), (jx+hw, jy)],  # 북
            [(jx-hw, jy),    (jx-hw, jy+ln), (jx+hw, jy+ln), (jx+hw, jy)],  # 남
            [(jx,    jy-hw), (jx+ln, jy-hw), (jx+ln, jy+hw), (jx,    jy+hw)],  # 동
            [(jx,    jy-hw), (jx-ln, jy-hw), (jx-ln, jy+hw), (jx,    jy+hw)],  # 서
        ]
        # 교차로 중앙 박스
        arms.append([(jx-hw, jy-hw), (jx+hw, jy-hw),
                     (jx+hw, jy+hw), (jx-hw, jy+hw)])

        for corners in arms:
            pts = [self._proj(wx, wy, 0.0) for wx, wy in corners]
            if all(p is not None for p in pts):
                cv2.fillPoly(img, [np.array(pts, np.int32)], self._ROAD)

        # 차선 중앙선 (각 암 중앙에 흰 점선)
        dash_segs = [
            [(jx, jy),      (jx, jy - ln)],   # 북
            [(jx, jy),      (jx, jy + ln)],   # 남
            [(jx, jy),      (jx + ln, jy)],   # 동
            [(jx, jy),      (jx - ln, jy)],   # 서
        ]
        for (x1, y1), (x2, y2) in dash_segs:
            steps = 12
            for k in range(steps):
                t0, t1 = k / steps, (k + 0.45) / steps
                p0 = self._proj(x1 + (x2-x1)*t0, y1 + (y2-y1)*t0, 0.02)
                p1 = self._proj(x1 + (x2-x1)*t1, y1 + (y2-y1)*t1, 0.02)
                if self._in_canvas(p0) and self._in_canvas(p1):
                    cv2.line(img, p0, p1, self._LANE, 1, cv2.LINE_AA)

        # 교차로 중심 강조 점
        c_pt = self._proj(jx, jy, 0.0)
        if self._in_canvas(c_pt):
            cv2.circle(img, c_pt, 4, self._CENTER, -1, cv2.LINE_AA)

    # ── 거리 원호 ──────────────────────────────
    def _draw_distance_arcs(self, img: np.ndarray):
        jx, jy = self.junc.x, self.junc.y
        for r_m in [10, 20, 30, 40]:
            pts = []
            for deg in range(0, 361, 4):
                a  = math.radians(deg)
                wx = jx + r_m * math.cos(a)
                wy = jy + r_m * math.sin(a)
                p  = self._proj(wx, wy, 0.0)
                if self._in_canvas(p):
                    pts.append(p)
            for i in range(len(pts) - 1):
                cv2.line(img, pts[i], pts[i+1], self._GRID, 1)
            # 동쪽 레이블
            lp = self._proj(jx + r_m + 1, jy, 0.0)
            if self._in_canvas(lp):
                cv2.putText(img, f"{r_m}m", (lp[0]+2, lp[1]-2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.26, (55,65,80), 1)

    # ── 3D 차량 박스 ───────────────────────────
    def _draw_vehicle_box(self, img: np.ndarray, actor,
                          vtype: str, speed: float):
        """CARLA bounding box를 3D 원근 투영으로 그림"""
        color     = COLORS.get(vtype, COLORS["unknown"])
        dark_col  = tuple(max(0, c - 80) for c in color)
        loc       = actor.get_location()
        yaw_rad   = math.radians(actor.get_transform().rotation.yaw)
        bb        = actor.bounding_box
        hl, hw, hh = bb.extent.x, bb.extent.y, bb.extent.z  # half-dims

        cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)

        # 로컬 → 월드: 8 꼭짓점
        # CARLA bounding box 로컬: X=forward, Y=right, Z=up
        local_pts = [
            (-hl, -hw, 0.0), (-hl,  hw, 0.0),
            ( hl,  hw, 0.0), ( hl, -hw, 0.0),
            (-hl, -hw, hh*2), (-hl,  hw, hh*2),
            ( hl,  hw, hh*2), ( hl, -hw, hh*2),
        ]
        def to_world(lx, ly, lz):
            wx = loc.x + lx * cos_y - ly * sin_y
            wy = loc.y + lx * sin_y + ly * cos_y
            wz = loc.z + lz
            return self._proj(wx, wy, wz)

        c = [to_world(*lp) for lp in local_pts]   # 0-3: bottom, 4-7: top

        # 화면 밖 검사
        if any(p is None for p in c):
            # 최소 표시: 중심에 작은 원
            center_pt = self._proj(loc.x, loc.y, hh)
            if self._in_canvas(center_pt):
                cv2.circle(img, center_pt, 5, color, -1, cv2.LINE_AA)
            return

        def draw_quad(pts_idx, fill_color, border_color=(255,255,255)):
            quad = np.array([c[i] for i in pts_idx], np.int32)
            if len(quad) == 4:
                cv2.fillPoly(img,  [quad], fill_color)
                cv2.polylines(img, [quad], True, border_color, 1, cv2.LINE_AA)

        # 바닥면 (매우 어둡게)
        draw_quad([0,1,2,3], tuple(max(0,x-120) for x in color), (40,45,55))

        # 측면 — 남쪽/동쪽 면만 그림 (보이는 면만)
        side_dark = tuple(max(0, x - 50) for x in color)
        draw_quad([0,4,5,1], side_dark, (0,0,0))   # 왼쪽 면
        draw_quad([3,7,4,0], side_dark, (0,0,0))   # 앞면

        # 세로 모서리 (4개)
        for bot, top in [(0,4),(1,5),(2,6),(3,7)]:
            if self._in_canvas(c[bot]) and self._in_canvas(c[top]):
                cv2.line(img, c[bot], c[top], color, 1, cv2.LINE_AA)

        # 윗면 (밝은 색상 — 가장 위에 그림)
        draw_quad([4,5,6,7], color)

        # 진행 방향 화살표 (윗면 중앙에서 앞으로)
        arrow_base = self._proj(loc.x, loc.y, hh * 2 + 0.2)
        fwd_wx = loc.x + (hl + 2.5) * cos_y
        fwd_wy = loc.y + (hl + 2.5) * sin_y
        arrow_tip = self._proj(fwd_wx, fwd_wy, hh * 2 + 0.2)
        if self._in_canvas(arrow_base) and self._in_canvas(arrow_tip):
            cv2.arrowedLine(img, arrow_base, arrow_tip,
                            (255, 255, 255), 1, cv2.LINE_AA, tipLength=0.4)

        # 라벨 (윗면 위 부동)
        label_pt = self._proj(loc.x, loc.y, hh * 2 + 1.8)
        if self._in_canvas(label_pt):
            lx, ly = label_pt
            tag = f"{vtype.upper()} {speed:.0f}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.30, 1)
            cv2.rectangle(img, (lx - tw//2 - 2, ly - th - 2),
                          (lx + tw//2 + 2, ly + 2),
                          tuple(max(0, c - 100) for c in color), -1)
            cv2.putText(img, tag, (lx - tw//2, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (255, 255, 255), 1, cv2.LINE_AA)

    # ── CCTV 마커 ──────────────────────────────
    def _draw_cctv_marker(self, img: np.ndarray):
        cx, cy, cz = self.cam_loc.x, self.cam_loc.y, self.cam_loc.z
        base = self._proj(cx, cy, 0.0)
        top  = self._proj(cx, cy, cz)
        if self._in_canvas(base) and self._in_canvas(top):
            cv2.line(img, base, top, self._CCTV, 2, cv2.LINE_AA)
            cv2.circle(img, top, 6, self._CCTV, -1, cv2.LINE_AA)
            cv2.putText(img, "CCTV", (top[0]+5, top[1]-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, self._CCTV, 1)

    # ── V2I 신호선 ─────────────────────────────
    def _draw_v2i_lines(self, img: np.ndarray, results: list):
        """교차로 중심 → 각 차량 점선 V2I 신호선"""
        jx, jy = self.junc.x, self.junc.y
        c0 = self._proj(jx, jy, 0.5)
        if not self._in_canvas(c0):
            return
        for r in results:
            loc   = r["actor"].get_location()
            vtype = r["type"]
            c1    = self._proj(loc.x, loc.y, 0.5)
            if not self._in_canvas(c1):
                continue
            color = COLORS.get(vtype, COLORS["unknown"])
            # 점선 효과
            dx = c1[0] - c0[0]
            dy = c1[1] - c0[1]
            length = math.hypot(dx, dy)
            if length < 1:
                continue
            steps = max(int(length // 8), 1)
            for k in range(steps):
                t0 = k / steps
                t1 = (k + 0.45) / steps
                p0 = (int(c0[0] + dx * t0), int(c0[1] + dy * t0))
                p1 = (int(c0[0] + dx * t1), int(c0[1] + dy * t1))
                dim = tuple(max(0, c - 80) for c in color)
                cv2.line(img, p0, p1, dim, 1, cv2.LINE_AA)

    # ── 범례 ───────────────────────────────────
    def _draw_legend(self, img: np.ndarray, counts: dict):
        """우하단 반투명 패널에 차종별 카운트"""
        n     = len(CLASSES)
        pw    = 120
        ph    = n * 19 + 28
        px0   = MAP_PX - pw - 4
        py0   = MAP_PX - ph - 4

        # 반투명 배경
        roi = img[py0:py0+ph, px0:px0+pw].copy()
        cv2.addWeighted(roi, 0.35, np.zeros_like(roi), 0.65, 0, roi)
        img[py0:py0+ph, px0:px0+pw] = roi
        cv2.rectangle(img, (px0, py0), (px0+pw, py0+ph), (35,40,55), 1)

        cv2.putText(img, "V2I RADAR", (px0+6, py0+13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 180, 255), 1)
        for i, cls in enumerate(CLASSES):
            color = COLORS[cls]
            iy    = py0 + 22 + i * 19
            cv2.rectangle(img, (px0+6, iy-7), (px0+18, iy+4), color, -1)
            cv2.putText(img, f"{cls.upper()}  {counts.get(cls, 0)}",
                        (px0+22, iy+3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.31, color, 1)
        total = sum(counts.values())
        cv2.putText(img, f"TOTAL  {total}",
                    (px0+6, py0 + ph - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1)

    # ── 메인 렌더 ──────────────────────────────
    def render(self, results: list) -> np.ndarray:
        """
        Tesla 3D V2I 레이더 뷰 생성

        Args:
            results: list of dict { "actor", "type", "speed" }
        Returns:
            BGR 이미지 (MAP_PX × MAP_PX)
        """
        img = np.full((MAP_PX, MAP_PX, 3), self._BG, dtype=np.uint8)

        self._draw_distance_arcs(img)
        self._draw_road(img)
        self._draw_v2i_lines(img, results)
        self._draw_cctv_marker(img)

        # 깊이 정렬: 멀리 있는 차량 먼저 그려서 가까운 차량이 앞에 오도록
        def depth_key(r):
            loc = r["actor"].get_location()
            rel = np.array([loc.x - self._eye[0],
                            loc.y - self._eye[1],
                            loc.z - self._eye[2]])
            return float(np.dot(self._fwd, rel))

        counts = {c: 0 for c in CLASSES}
        for r in sorted(results, key=depth_key, reverse=True):
            self._draw_vehicle_box(img, r["actor"], r["type"], r["speed"])
            if r["type"] in counts:
                counts[r["type"]] += 1

        self._draw_legend(img, counts)
        return img


# =============================================
# V2I 관제 시스템 메인 클래스
# =============================================
class V2IMonitorSystem:
    """
    V2I 스마트 교차로 3D 관제 시스템

    파이프라인:
        world.tick()
        → CARLA actor list 취득
        → 카메라 투영 (world_to_pixel)으로 화면 내 차량 필터
        → CNN 분류 (CLASSIFY_INTERVAL 프레임마다 캐시 갱신)
        → 카메라 뷰 오버레이 (BBox + 차종·속도·거리)
        → 3D V2I 레이더 맵 렌더링
        → 합성 화면 출력
    """

    def __init__(self):
        # CARLA 연결
        self.client = carla.Client(CARLA_HOST, CARLA_PORT)
        self.client.set_timeout(15.0)
        self.world  = self.client.get_world()
        self._map_name = self.world.get_map().name.split("/")[-1]
        print(f"[V2I] CARLA 연결 완료 | 맵: {self._map_name}")

        # 동기 모드 활성화
        settings = self.world.get_settings()
        settings.synchronous_mode    = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

        # TrafficManager
        self.tm = self.client.get_trafficmanager(TM_PORT)
        self.tm.set_synchronous_mode(True)
        self.tm.set_global_distance_to_leading_vehicle(2.5)

        # 교차로 탐색
        self.junction_center = find_intersection(self.world)

        # CCTV 카메라 설치
        self.camera, self.img_queue, self.cam_tf = spawn_cctv_camera(
            self.world, self.junction_center
        )
        self.K = build_projection_matrix(IMG_W, IMG_H, CAM_FOV)

        # NPC 소환 전 TM 토폴로지 초기화 (10 tick)
        for _ in range(10):
            self.world.tick()
        self.npc_list = spawn_npc_vehicles(self.client, self.world, self.tm, NUM_NPC)

        # 워밍업 (카메라 & TM 안정화)
        print("[V2I] 워밍업 중... (60 tick)")
        for _ in range(60):
            self.world.tick()
        # 큐 초기화
        while not self.img_queue.empty():
            try:
                self.img_queue.get_nowait()
            except queue.Empty:
                break

        # CNN 분류기 + 캐시
        self.classifier       = VehicleClassifier()
        self._classify_cache: dict[int, str] = {}   # vehicle_id → class_str
        self._frame_idx       = 0

        # 3D 레이더 맵 렌더러 (Tesla 계기판 스타일)
        self.radar = V2IMapRenderer(
            junction_center=self.junction_center,
            cam_location=self.cam_tf.location,
        )

        # FPS 계측
        self._t_last = time.time()
        self._fps    = 0.0

        print("[V2I] 시스템 초기화 완료\n"
              "  ─ 조작: q=종료  r=NPC재소환")

    # ─────────────────────────────────────────────────────
    def _detect_and_classify(self, frame: np.ndarray) -> list:
        """
        CARLA actor list → 화면 내 차량 탐지 → CNN 분류

        Returns:
            list of dict:
              { "actor", "type", "bbox":(x1,y1,x2,y2), "speed" }
        """
        self._frame_idx += 1
        results = []

        for vehicle in self.world.get_actors().filter("vehicle.*"):
            vid   = vehicle.id
            bbox2 = get_vehicle_bbox_pixels(vehicle, self.camera, self.K, IMG_W, IMG_H)
            if bbox2 is None:
                continue

            x1, y1, x2, y2 = bbox2
            if (x2 - x1) < MIN_BBOX_PX or (y2 - y1) < MIN_BBOX_PX:
                continue

            # CNN 분류 (캐시 활용)
            if (self._frame_idx % CLASSIFY_INTERVAL == 0
                    or vid not in self._classify_cache):
                crop  = frame[y1:y2, x1:x2]
                vtype = self.classifier.classify(crop)
                self._classify_cache[vid] = vtype
            else:
                vtype = self._classify_cache[vid]

            # 속도 (m/s → km/h)
            v     = vehicle.get_velocity()
            speed = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)

            results.append({
                "actor": vehicle,
                "type":  vtype,
                "bbox":  bbox2,
                "speed": speed,
            })

        return results

    # ─────────────────────────────────────────────────────
    def _draw_camera_view(self, frame: np.ndarray, results: list) -> np.ndarray:
        """카메라 영상 위에 BBox · 차종 · 속도 · 거리 오버레이"""

        for r in results:
            x1, y1, x2, y2 = r["bbox"]
            vtype = r["type"]
            speed = r["speed"]
            color = COLORS.get(vtype, COLORS["unknown"])

            # 거리 계산
            loc  = r["actor"].get_location()
            dist = math.sqrt(
                (loc.x - self.cam_tf.location.x) ** 2 +
                (loc.y - self.cam_tf.location.y) ** 2 +
                (loc.z - self.cam_tf.location.z) ** 2
            )

            # BBox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 레이블 배경
            label = f"{vtype.upper()} | {speed:4.0f}km/h | {dist:4.1f}m"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
            ly = max(y1 - 6, lh + 4)
            cv2.rectangle(frame, (x1, ly - lh - 3), (x1 + lw + 6, ly + 2), color, -1)
            cv2.putText(frame, label, (x1 + 3, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 0, 0), 1, cv2.LINE_AA)

        # ── HUD 상단 ──────────────────────────────────────────
        hud_bg = frame.copy()
        cv2.rectangle(hud_bg, (0, 0), (IMG_W, 42), (0, 0, 0), -1)
        cv2.addWeighted(hud_bg, 0.55, frame, 0.45, 0, frame)

        title = (f"V2I Smart Intersection Monitor  |  "
                 f"{self._map_name}  |  FPS {self._fps:.1f}")
        cv2.putText(frame, title, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, title, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 190, 255), 1, cv2.LINE_AA)

        # ── HUD 우상단: 차종별 카운트 ─────────────────────────
        counts = {c: 0 for c in CLASSES}
        for r in results:
            if r["type"] in counts:
                counts[r["type"]] += 1

        panel_x = IMG_W - 130
        cv2.rectangle(frame, (panel_x - 6, 46), (IMG_W - 4, 46 + len(CLASSES)*22 + 24),
                      (0, 0, 0), -1)
        for i, cls in enumerate(CLASSES):
            txt = f"{cls.upper()}: {counts[cls]}"
            cv2.putText(frame, txt, (panel_x, 64 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, COLORS[cls], 2, cv2.LINE_AA)
        cv2.putText(frame, f"TOTAL: {len(results)}",
                    (panel_x, 64 + len(CLASSES) * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)

        return frame

    # ─────────────────────────────────────────────────────
    def _compose_display(self, cam_frame: np.ndarray, radar: np.ndarray) -> np.ndarray:
        """카메라 뷰 우측 하단에 레이더 맵 합성"""
        h, w   = cam_frame.shape[:2]
        mh, mw = radar.shape[:2]
        margin = 12
        border = 2

        # 파란 테두리 프레임
        framed = cv2.copyMakeBorder(radar, border, border, border, border,
                                    cv2.BORDER_CONSTANT, value=(0, 180, 255))
        fh, fw = framed.shape[:2]

        ys = h - fh - margin
        xs = w - fw - margin

        # 반투명 배경
        roi = cam_frame[ys:ys+fh, xs:xs+fw].copy()
        cv2.addWeighted(roi, 0.25, np.zeros_like(roi), 0.75, 0, roi)
        cam_frame[ys:ys+fh, xs:xs+fw] = roi

        cam_frame[ys:ys+fh, xs:xs+fw] = framed

        # 맵 제목
        cv2.putText(cam_frame, "[ 3D V2I RADAR MAP ]",
                    (xs, ys - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 180, 255), 1, cv2.LINE_AA)

        return cam_frame

    # ─────────────────────────────────────────────────────
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """단일 프레임 전체 파이프라인 처리"""
        results = self._detect_and_classify(frame)
        display = self._draw_camera_view(frame.copy(), results)
        radar   = self.radar.render(results)
        display = self._compose_display(display, radar)

        # FPS
        now          = time.time()
        self._fps    = 1.0 / max(now - self._t_last, 1e-6)
        self._t_last = now

        return display

    # ─────────────────────────────────────────────────────
    def _respawn_npcs(self):
        """NPC 차량 전체 제거 후 재소환 (r 키 입력 시)"""
        print("[V2I] NPC 재소환 중...")
        for v in self.npc_list:
            try:
                if v.is_alive:
                    v.destroy()
            except Exception:
                pass
        self.world.tick()
        self.npc_list = spawn_npc_vehicles(self.client, self.world, self.tm, NUM_NPC)
        self._classify_cache.clear()

    # ─────────────────────────────────────────────────────
    def run(self):
        """메인 관제 루프"""
        cv2.namedWindow("V2I Smart Intersection Monitor", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("V2I Smart Intersection Monitor", IMG_W, IMG_H)
        print("[V2I] 관제 시작  ─  q: 종료  |  r: NPC 재소환")

        while True:
            self.world.tick()

            # 프레임 취득 (sync 모드이므로 tick 직후 바로 사용 가능)
            try:
                raw = self.img_queue.get_nowait()
            except queue.Empty:
                continue

            arr = np.frombuffer(raw.raw_data, np.uint8).reshape(
                (raw.height, raw.width, 4)
            )[:, :, :3].copy()   # BGRA → BGR

            display = self.process_frame(arr)
            cv2.imshow("V2I Smart Intersection Monitor", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                self._respawn_npcs()

    # ─────────────────────────────────────────────────────
    def cleanup(self):
        """CARLA 리소스 정리"""
        print("[V2I] 정리 중...")

        # 액터 먼저 제거
        try:
            self.camera.destroy()
        except Exception:
            pass

        batch = [carla.command.DestroyActor(v)
                 for v in self.npc_list if v.is_alive]
        if batch:
            self.client.apply_batch_sync(batch, True)

        # Sync 모드 해제 (MEMORY.md 주의사항: apply_settings가 exit() 유발 가능)
        try:
            for _ in range(3):
                self.world.tick()
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
        except Exception:
            pass   # CARLA가 exit()를 내부 호출해도 여기서 정상 처리

        cv2.destroyAllWindows()
        print("[V2I] 종료 완료")


# =============================================
# 진입점
# =============================================
def main():
    system = V2IMonitorSystem()
    try:
        system.run()
    finally:
        system.cleanup()


if __name__ == "__main__":
    main()
