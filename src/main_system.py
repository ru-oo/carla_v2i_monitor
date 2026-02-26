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
CAM_HEIGHT = 20.0       # V2I 관제 카메라 고도 (m) — 20m 부감 시야
CAM_OFFSET = 6.0        # 교차로 중심 대비 수평 오프셋 (m, SW 방향)
NUM_NPC    = 40         # NPC 차량 수

MAP_PX     = 500        # 레이더 맵 픽셀 크기
MAP_RANGE  = 100.0      # 맵 표시 범위 (±50 m)
MAP_SCALE  = MAP_PX / MAP_RANGE   # 5 px / m

CLASSIFY_INTERVAL = 5   # N프레임마다 CNN 재분류 (CPU 부하 완화)
MIN_BBOX_PX       = 12  # 유효 BBox 최소 변 길이 (px)

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
    V2I 관제용 고고도 부감 카메라 설치

    data_collector.py의 신호등 폴(6 m) 방식 대신
    교차로 중심에서 SW 방향 CAM_OFFSET m, 높이 CAM_HEIGHT m 에 배치.
    → pitch ≈ -73° 부감 시야로 교차로 전체를 포착.

    Returns:
        (camera_actor, img_queue, cam_transform)
    """
    bp_lib = world.get_blueprint_library()
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(IMG_W))
    cam_bp.set_attribute("image_size_y", str(IMG_H))
    cam_bp.set_attribute("fov", str(CAM_FOV))

    # 교차로 중심에서 SW 방향으로 CAM_OFFSET m 수평 이동 + 고도 CAM_HEIGHT m
    # SW: x -= offset * cos(45°) = offset * 0.7071
    #     y += offset * sin(45°) = offset * 0.7071
    off = CAM_OFFSET * 0.7071
    cx  = junc_center.x - off
    cy  = junc_center.y + off
    cz  = junc_center.z + CAM_HEIGHT

    dx    = junc_center.x - cx          # +off (동쪽)
    dy    = junc_center.y - cy          # -off (북쪽)
    dz    = junc_center.z - cz          # -CAM_HEIGHT
    horiz = math.sqrt(dx * dx + dy * dy)  # ≈ CAM_OFFSET

    yaw   = math.degrees(math.atan2(dy, dx))
    pitch = math.degrees(math.atan2(dz, horiz))   # ≈ -73° (CAM_HEIGHT=20, offset=6)

    cam_transform = carla.Transform(
        carla.Location(x=cx, y=cy, z=cz),
        carla.Rotation(pitch=pitch, yaw=yaw, roll=0.0),
    )

    img_q  = queue.Queue(maxsize=2)
    camera = world.spawn_actor(cam_bp, cam_transform)
    camera.listen(lambda img: img_q.put_nowait(img) if not img_q.full() else None)

    print(f"[V2I] CCTV 설치 완료\n"
          f"  위치: ({cx:.1f}, {cy:.1f}, {cz:.1f}m)"
          f"  pitch={pitch:.1f}°  yaw={yaw:.1f}°  horiz={horiz:.1f}m")
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
    Top-view 3D 레이더 맵 (500 × 500 픽셀, 100 m × 100 m 범위)

    시각 요소:
        - 배경 격자 + 거리 원 (10 / 30 / 50 m)
        - 교차로 십자 도로
        - 카메라 위치 삼각형 마커
        - V2I 신호선 (카메라 → 각 차량)
        - 차종별 색상 원 + 속도 텍스트
        - 범례 + 차종별 카운트
    """

    def __init__(self, junction_center: carla.Location, cam_location: carla.Location):
        self.center  = junction_center
        self.cam_loc = cam_location

    # ── 좌표 변환 ──────────────────────────────
    def _to_px(self, world_x: float, world_y: float) -> tuple[int, int]:
        """월드 XY → 맵 픽셀 (CARLA Y=South=화면 아래 방향과 일치)"""
        dx = world_x - self.center.x
        dy = world_y - self.center.y
        px = int(MAP_PX / 2 + dx * MAP_SCALE)
        py = int(MAP_PX / 2 + dy * MAP_SCALE)
        return px, py

    # ── 배경 ───────────────────────────────────
    def _draw_grid(self, img: np.ndarray):
        step = int(10 * MAP_SCALE)   # 10 m = 50 px
        for i in range(0, MAP_PX, step):
            cv2.line(img, (i, 0), (i, MAP_PX - 1), (28, 28, 28), 1)
            cv2.line(img, (0, i), (MAP_PX - 1, i), (28, 28, 28), 1)
        c = MAP_PX // 2
        for m, clr in [(10, (40,40,40)), (30, (40,40,40)), (50, (50,50,50))]:
            r = int(m * MAP_SCALE)
            cv2.circle(img, (c, c), r, clr, 1)
            cv2.putText(img, f"{m}m", (c + r + 2, c - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.27, (70, 70, 70), 1)

    def _draw_road_cross(self, img: np.ndarray):
        road_w = int(MAP_PX * 0.18)
        c      = MAP_PX // 2
        cv2.rectangle(img, (c - road_w//2, 0),    (c + road_w//2, MAP_PX),  (55,55,55), -1)
        cv2.rectangle(img, (0, c - road_w//2),     (MAP_PX, c + road_w//2), (55,55,55), -1)
        for i in range(0, MAP_PX, 20):
            cv2.line(img, (c, i), (c, min(i+10, MAP_PX-1)), (0, 160, 160), 1)
            cv2.line(img, (i, c), (min(i+10, MAP_PX-1), c), (0, 160, 160), 1)

    def _draw_camera_marker(self, img: np.ndarray):
        cx, cy = self._to_px(self.cam_loc.x, self.cam_loc.y)
        if 0 <= cx < MAP_PX and 0 <= cy < MAP_PX:
            pts = np.array([[cx, cy-11], [cx-8, cy+7], [cx+8, cy+7]], np.int32)
            cv2.fillPoly(img, [pts], (0, 210, 255))
            cv2.polylines(img, [pts], True, (255,255,255), 1)
            cv2.putText(img, "CAM", (cx - 13, cy + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 210, 255), 1)

    def _draw_legend(self, img: np.ndarray, counts: dict):
        y0 = 10
        for i, cls in enumerate(CLASSES):
            color = COLORS[cls]
            iy    = y0 + i * 19
            cv2.circle(img, (12, iy), 6, color, -1)
            cv2.putText(img, f"{cls.upper()}  {counts.get(cls, 0)}",
                        (22, iy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1)
        total = sum(counts.values())
        ty    = y0 + len(CLASSES) * 19 + 6
        cv2.putText(img, f"TOTAL  {total}", (8, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 200, 200), 1)

    # ── 메인 렌더 ──────────────────────────────
    def render(self, results: list) -> np.ndarray:
        """
        Args:
            results: list of dict
                { "actor": carla.Actor, "type": str, "speed": float }
        Returns:
            BGR 레이더 맵 이미지 (MAP_PX × MAP_PX)
        """
        img = np.zeros((MAP_PX, MAP_PX, 3), dtype=np.uint8)
        self._draw_grid(img)
        self._draw_road_cross(img)

        # V2I 신호선 (차량 원 뒤에 깔리도록 먼저 그림)
        cam_px = self._to_px(self.cam_loc.x, self.cam_loc.y)
        for r in results:
            loc = r["actor"].get_location()
            vx, vy = self._to_px(loc.x, loc.y)
            if 0 <= vx < MAP_PX and 0 <= vy < MAP_PX:
                color = COLORS.get(r["type"], COLORS["unknown"])
                cv2.line(img, cam_px, (vx, vy), color, 1, cv2.LINE_AA)

        # 카메라 마커
        self._draw_camera_marker(img)

        # 차량 원
        counts = {c: 0 for c in CLASSES}
        for r in results:
            loc   = r["actor"].get_location()
            vtype = r["type"]
            speed = r["speed"]
            vx, vy = self._to_px(loc.x, loc.y)
            if 0 <= vx < MAP_PX and 0 <= vy < MAP_PX:
                color = COLORS.get(vtype, COLORS["unknown"])
                cv2.circle(img, (vx, vy), 9,  color,         -1)
                cv2.circle(img, (vx, vy), 9,  (255,255,255),  1)
                cv2.putText(img, ICON.get(vtype, "?"),
                            (vx - 4, vy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.37, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(img, f"{speed:.0f}",
                            (vx + 11, vy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1)
                if vtype in counts:
                    counts[vtype] += 1

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

        # 레이더 맵 렌더러
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
