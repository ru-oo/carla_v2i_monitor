"""
data_collector.py
=================
담당: 김예진 (CARLA 시뮬레이션 & 데이터 엔지니어링)
브랜치: feature/carla-sim

역할:
    - CARLA 시뮬레이터에서 교차로 코너 CCTV 카메라 배치 (실제 CCTV 위치 재현)
    - 날씨/교통량 시나리오 설정 및 시뮬레이션 구동
    - Ground Truth 기반 차량 이미지 자동 캡처 및 저장
    - CNN 학습용 데이터셋 생성
    - CARLA 화면에 실시간 디버그 뷰(BBox + 라벨) 표시

산출물:
    - data/vehicle_images/car/   ← 승용차 크롭 이미지
    - data/vehicle_images/truck/ ← 트럭 크롭 이미지
    - data/raw/                  ← 원본 프레임 저장 (선택)

실행 방법:
    1. CARLA 서버 먼저 실행: ./CarlaUE4.sh  (또는 CarlaUE4.exe)
    2. python src/data_collector.py
"""

# test_git
import carla
import random
import time
import os
import cv2
import numpy as np
import queue
import math

# =============================================
# 설정값 (Config)
# =============================================
CARLA_HOST = "localhost"
CARLA_PORT = 2000

# --- CCTV 카메라 설정 (실제 교차로 신호등 폴 CCTV) ---
# 배치 전략:
#   1. junction.get_waypoints()로 SW 진입로 도로 위 좌표 확보 (건물 내부 방지)
#   2. 해당 위치에서 CAMERA_HEIGHT(6m) 올려 신호등 폴 높이 재현
#   3. 교차로 중심까지 yaw / pitch 를 수학적으로 자동 계산
CAMERA_HEIGHT    = 6.0          # 실제 신호등 CCTV 폴 높이 (m)
# pitch / yaw / offset 은 spawn_cctv_camera() 에서 동적 계산
MAX_VEHICLE_DIST = 60.0         # 이 거리(m) 초과 차량은 수집 제외 (너무 작아 품질 저하)

IMG_WIDTH     = 1280
IMG_HEIGHT    = 720
FOV           = 90.0            # 85 → 90 (교차로 전체가 시야에 들어오도록)

OUTPUT_DIR    = "data/vehicle_images"
RAW_DIR       = "data/raw"
SAVE_RAW      = False           # 원본 프레임도 저장할지 여부

# --- BBox 필터링 ---
MIN_BBOX_PX      = 25           # 30 → 25 (CCTV 뷰에서 멀리 보이는 차량도 수집)
MAX_BBOX_PX      = 500          # 400 → 500 (트럭 전체가 한 장에 들어오도록)
CAPTURE_COOLDOWN = 0.5          # 1.0 → 0.5초 (더 많은 각도/장면 수집)

# --- 시뮬레이션 규모 ---
NUM_NPC       = 80              # 40 → 80 (차량 수 증가)
SIM_DURATION  = 300             # 180 → 300초 (더 많은 데이터 수집)

# 화면 표시용 색상 (BGR)
COLOR_CAR    = (0, 255, 0)      # 초록
COLOR_TRUCK  = (0, 100, 255)    # 주황
COLOR_OTHER  = (200, 200, 200)

# 트럭 계열 블루프린트 키워드
TRUCK_KEYWORDS = ["truck", "van", "bus", "ambulance", "firetruck", "carlacola",
                  "sprinter", "t2", "cybertruck", "european_hgv"]

# 수집 제외 키워드 (이륜차 / 트레일러)
# - 이륜차: 카메라 시야에서 나무처럼 보이거나 불필요한 샘플 생성
# - trailer: 트럭 차체와 별도 액터로 생성되어 분할 수집 발생
EXCLUDE_VEHICLE_KEYWORDS = [
    "bike", "cycle", "vespa", "harley", "kawasaki",
    "yamaha", "ninja", "omafiets", "crossbike", "century",
    "trailer",
]


# =============================================
# 유틸 함수
# =============================================
def is_truck(blueprint_id: str) -> bool:
    """블루프린트 ID로 트럭/대형차 여부 판단"""
    bp_lower = blueprint_id.lower()
    return any(kw in bp_lower for kw in TRUCK_KEYWORDS)


def is_valid_vehicle(blueprint_id: str) -> bool:
    """
    데이터 수집에 적합한 4륜 차량인지 확인

    제외 대상:
        - 이륜차(bicycle / motorcycle): 나무처럼 보이거나 작아서 잡음 데이터 생성
        - trailer: 트럭과 별개 액터로 생성되어 동일 트럭이 분할 저장되는 원인
    """
    bp_lower = blueprint_id.lower()
    return not any(kw in bp_lower for kw in EXCLUDE_VEHICLE_KEYWORDS)


def get_vehicle_label(blueprint_id: str) -> str:
    return "truck" if is_truck(blueprint_id) else "car"


def setup_output_dirs():
    """학습 데이터 저장 폴더 생성"""
    for sub in ["car", "truck"]:
        os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)
    if SAVE_RAW:
        os.makedirs(RAW_DIR, exist_ok=True)
    print(f"[DataCollector] 출력 폴더 준비 완료: {OUTPUT_DIR}")


def connect_to_carla():
    """CARLA 서버에 연결"""
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(15.0)
    world = client.get_world()
    print(f"[DataCollector] CARLA 연결 성공: {world.get_map().name}")
    return client, world


# =============================================
# 카메라 & 인프라 설치
# =============================================
def find_intersection(world):
    """
    맵에서 실제 도로 교차로(junction) 위치 탐색

    CARLA waypoint API의 is_junction 플래그를 사용해
    spawn_point 평균(건물/공원으로 빠질 수 있음) 대신
    실제 도로 위 junction 중심을 반환한다.

    선택 기준: 맵 전체 junction 평균 위치에 가장 가까운 junction
              → 도심 중앙의 대표 교차로
    """
    carla_map = world.get_map()

    # 5m 간격 waypoint 생성 후 junction 필터링
    all_wps = carla_map.generate_waypoints(5.0)
    junction_wps = [wp for wp in all_wps if wp.is_junction]

    if not junction_wps:
        # fallback: spawn point 평균
        print("[DataCollector] 경고: junction 없음 → spawn point 평균 사용")
        sps = carla_map.get_spawn_points()
        xs = [sp.location.x for sp in sps]
        ys = [sp.location.y for sp in sps]
        return carla.Location(x=sum(xs) / len(xs), y=sum(ys) / len(ys), z=0.0)

    # 모든 junction waypoint의 평균 위치 계산 (도심 중앙 기준점)
    cx = sum(wp.transform.location.x for wp in junction_wps) / len(junction_wps)
    cy = sum(wp.transform.location.y for wp in junction_wps) / len(junction_wps)

    # 평균 위치에 가장 가까운 junction waypoint 선택
    best_wp = min(
        junction_wps,
        key=lambda wp: (wp.transform.location.x - cx) ** 2
                     + (wp.transform.location.y - cy) ** 2,
    )
    loc = best_wp.transform.location
    print(f"[DataCollector] 도로 junction 탐색 완료: "
          f"({loc.x:.1f}, {loc.y:.1f}, z={loc.z:.1f})  "
          f"(전체 junction wp {len(junction_wps)}개 중 선택)")
    return carla.Location(x=loc.x, y=loc.y, z=loc.z)


def find_cctv_mount_position(world, junction_center):
    """
    교차로 SW 진입로의 도로 위 좌표 탐색 (건물 내부 스폰 방지 핵심 함수)

    고정 오프셋 대신 CARLA junction API 로 실제 도로 위 waypoint 를 반환.
    junction.get_waypoints() 가 반환하는 waypoint 는 항상 도로 위에 있음.

    선택 기준: junction 진입로 중 교차로 중심 대비 가장 SW 방향인 waypoint
               SW 점수 = (중심.x - 위치.x) + (위치.y - 중심.y)
                          ↑서쪽일수록 큰 값    ↑남쪽일수록 큰 값
    """
    carla_map = world.get_map()

    # junction_center 부근의 waypoint 에서 junction 객체 가져오기
    junction = None
    search_offsets = [(0,0),(3,0),(-3,0),(0,3),(0,-3),(5,5),(-5,5),(5,-5),(-5,-5)]
    for ox, oy in search_offsets:
        test_loc = carla.Location(x=junction_center.x + ox,
                                  y=junction_center.y + oy,
                                  z=junction_center.z)
        wp = carla_map.get_waypoint(test_loc, project_to_road=True)
        if wp and wp.is_junction:
            junction = wp.get_junction()
            break

    if junction is None:
        # fallback: 교차로 중심에서 SW 방향으로 snap-to-road
        sw_loc = carla.Location(x=junction_center.x - 15,
                                y=junction_center.y + 15,
                                z=junction_center.z)
        snap_wp = carla_map.get_waypoint(sw_loc, project_to_road=True)
        fallback = snap_wp.transform.location if snap_wp else junction_center
        print(f"[DataCollector] junction 없음 → fallback 위치 사용: "
              f"({fallback.x:.1f}, {fallback.y:.1f})")
        return fallback

    # junction 진입로 waypoint 중 가장 SW 방향 선택
    best_loc   = None
    best_score = -float("inf")
    for entry_wp, _ in junction.get_waypoints(carla.LaneType.Driving):
        loc = entry_wp.transform.location
        # SW 점수: 서쪽(x 감소) + 남쪽(y 증가) 합
        score = (junction_center.x - loc.x) + (loc.y - junction_center.y)
        if score > best_score:
            best_score = score
            best_loc   = loc

    if best_loc is None:
        best_loc = junction_center

    print(f"[DataCollector] CCTV 도로 마운트 위치: "
          f"({best_loc.x:.1f}, {best_loc.y:.1f}, z={best_loc.z:.1f})")
    return best_loc


def spawn_cctv_camera(world, junction_center):
    """
    교차로 SW 진입로 도로 위 신호등 폴에 실제 CCTV 배치

    배치 방식:
        1. find_cctv_mount_position() 으로 SW 진입로 도로 위 좌표 확보
           → 고정 오프셋 사용 시 건물 내부 스폰 문제 완전 해결
        2. 해당 좌표 + CAMERA_HEIGHT(6m) = 실제 신호등 CCTV 폴 높이
        3. 교차로 중심까지 yaw / pitch 를 math.atan2 로 정확히 계산
           → 어떤 맵, 어떤 교차로에서도 교차로 정중앙을 향함

    Returns:
        camera_sensor, image_queue
    """
    bp_lib = world.get_blueprint_library()
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(IMG_WIDTH))
    cam_bp.set_attribute("image_size_y", str(IMG_HEIGHT))
    cam_bp.set_attribute("fov", str(FOV))

    # 도로 위 마운트 좌표 (건물 내부 방지)
    mount = find_cctv_mount_position(world, junction_center)
    cam_x = mount.x
    cam_y = mount.y
    cam_z = mount.z + CAMERA_HEIGHT

    # 카메라 → 교차로 중심 벡터로 yaw / pitch 자동 계산
    dx = junction_center.x - cam_x
    dy = junction_center.y - cam_y
    dz = junction_center.z - cam_z          # 음수: 교차로는 카메라보다 아래
    horiz_dist = math.sqrt(dx * dx + dy * dy)
    yaw   = math.degrees(math.atan2(dy, dx))
    pitch = math.degrees(math.atan2(dz, horiz_dist))

    cam_transform = carla.Transform(
        carla.Location(x=cam_x, y=cam_y, z=cam_z),
        carla.Rotation(pitch=pitch, yaw=yaw, roll=0.0),
    )

    img_queue = queue.Queue(maxsize=5)
    camera = world.spawn_actor(cam_bp, cam_transform)
    if camera is None:
        raise RuntimeError(
            f"[DataCollector] 카메라 스폰 실패 — 위치: "
            f"({cam_x:.1f}, {cam_y:.1f}, {cam_z:.1f})"
        )
    camera.listen(lambda img: img_queue.put(img) if not img_queue.full() else None)

    print(f"[DataCollector] 교차로 CCTV 설치 완료\n"
          f"  위치  : ({cam_x:.1f}, {cam_y:.1f}, {cam_z:.1f}m)\n"
          f"  pitch : {pitch:.1f}°   yaw : {yaw:.1f}°   FOV : {FOV}°")
    return camera, img_queue


def set_weather(world, preset="clear"):
    """날씨 설정"""
    presets = {
        "clear":  carla.WeatherParameters.ClearNoon,
        "rain":   carla.WeatherParameters.HardRainNoon,
        "night":  carla.WeatherParameters.ClearNight,
        "cloudy": carla.WeatherParameters.CloudyNoon,
    }
    world.set_weather(presets.get(preset, carla.WeatherParameters.ClearNoon))
    print(f"[DataCollector] 날씨: {preset}")


def spawn_npc_vehicles(client, world, num=NUM_NPC):
    """
    NPC 차량 자동 생성 및 자율주행 활성화

    변경사항:
        - 이륜차(bike/cycle/motorcycle) 블루프린트 완전 제외
        - trailer 제외 (트럭 분할 방지)
        - 승용차:트럭 = 7:3 비율 유지
    """
    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    # 4륜 유효 차량만 필터링 (이륜차 / 트레일러 제외)
    valid_bps = [b for b in bp_lib.filter("vehicle.*") if is_valid_vehicle(b.id)]
    truck_bps = [b for b in valid_bps if is_truck(b.id)]
    car_bps   = [b for b in valid_bps if not is_truck(b.id)]

    print(f"[DataCollector] 유효 차량 블루프린트 — car: {len(car_bps)}종  truck: {len(truck_bps)}종")

    vehicles = []
    tm = client.get_trafficmanager(8000)
    tm.set_global_distance_to_leading_vehicle(2.0)

    for sp in spawn_points[:num]:
        # 승용차:트럭 = 7:3 비율
        if random.random() < 0.3 and truck_bps:
            bp = random.choice(truck_bps)
        elif car_bps:
            bp = random.choice(car_bps)
        else:
            bp = random.choice(valid_bps)

        actor = world.try_spawn_actor(bp, sp)
        if actor:
            actor.set_autopilot(True, tm.get_port())
            vehicles.append(actor)

    print(f"[DataCollector] NPC 차량 {len(vehicles)}대 생성 (목표 {num}대)")
    return vehicles


# =============================================
# Ground Truth → 픽셀 좌표 변환
# =============================================
def is_vehicle_visible(world, camera_location, vehicle):
    """
    카메라에서 차량까지 레이캐스트로 가시성 확인 (벽 너머 차량 필터링)

    원리:
        world.cast_ray(카메라 위치, 차량 중심) 으로 중간 장애물 검출.
        첫 번째 히트가 차량까지 거리의 85% 미만이면 벽/건물에 가려진 것.

    Returns:
        True  → 차량이 카메라에서 보임 (수집 OK)
        False → 벽/건물에 가려짐 (수집 제외)
    """
    veh_loc = vehicle.get_location()

    # 거리 사전 필터: MAX_VEHICLE_DIST 초과 시 제외 (cast_ray 호출 절약)
    if camera_location.distance(veh_loc) > MAX_VEHICLE_DIST:
        return False

    try:
        hits = world.cast_ray(camera_location, veh_loc)
    except Exception:
        # cast_ray 미지원 버전 → 필터링 없이 통과
        return True

    if not hits:
        return True  # 중간 장애물 없음

    veh_dist       = camera_location.distance(veh_loc)
    first_hit_dist = camera_location.distance(hits[0].location)

    # 첫 히트가 차량까지 거리의 85% 미만이면 차량 앞에 장애물 존재
    return first_hit_dist >= veh_dist * 0.85


def build_projection_matrix(w, h, fov):
    """
    카메라 내부 행렬(Intrinsic Matrix) 계산
    CARLA 공식 문서 방식
    """
    focal = w / (2.0 * math.tan(math.radians(fov) / 2.0))
    K = np.array([
        [focal,   0,    w / 2.0],
        [0,     focal,  h / 2.0],
        [0,       0,       1.0 ],
    ])
    return K


def world_to_pixel(world_loc, camera_actor, K):
    """
    CARLA 월드 좌표 → 카메라 픽셀 좌표 변환

    Args:
        world_loc : carla.Location  (차량 중심)
        camera_actor : 카메라 센서 액터
        K : 내부 행렬 (3x3)

    Returns:
        (u, v) 픽셀 좌표  또는  None (카메라 뒤쪽)
    """
    cam_transform = camera_actor.get_transform()

    # 월드 → 카메라 좌표계 변환
    # CARLA는 Left-Handed (UE4) 좌표계를 사용
    cam_inv = np.array(cam_transform.get_inverse_matrix())  # 4x4

    # 월드 포인트를 동차 좌표로
    world_pt = np.array([world_loc.x, world_loc.y, world_loc.z, 1.0])

    # 카메라 좌표계로
    cam_pt = cam_inv @ world_pt   # (4,)
    x_cam, y_cam, z_cam = cam_pt[0], cam_pt[1], cam_pt[2]

    # UE4 → 카메라 좌표계
    # CARLA 공식: forward=x, right=y, up=z  →  image: u=y, v=z
    x_img = y_cam
    y_img = -z_cam
    z_img = x_cam

    if z_img <= 0:
        return None  # 카메라 뒤에 있음

    # 투영
    u = K[0, 0] * (x_img / z_img) + K[0, 2]
    v = K[1, 1] * (y_img / z_img) + K[1, 2]

    return int(u), int(v)


def get_vehicle_bbox_pixels(vehicle, camera_actor, K, img_w, img_h):
    """
    차량의 3D Bounding Box 8개 꼭짓점을 픽셀에 투영하여
    2D BBox (x, y, w, h) 반환

    Returns:
        (x, y, w, h) 또는 None
    """
    bbox_3d  = vehicle.bounding_box
    transform = vehicle.get_transform()

    verts = bbox_3d.get_world_vertices(transform)

    pixels = []
    for v in verts:
        px = world_to_pixel(v, camera_actor, K)
        if px is not None:
            pu, pv = px
            if 0 <= pu < img_w and 0 <= pv < img_h:
                pixels.append((pu, pv))

    if len(pixels) < 2:
        return None

    us = [p[0] for p in pixels]
    vs = [p[1] for p in pixels]
    x1, y1 = max(0, min(us)), max(0, min(vs))
    x2, y2 = min(img_w - 1, max(us)), min(img_h - 1, max(vs))

    w = x2 - x1
    h = y2 - y1

    # 너무 작거나 큰 박스 제외
    if w < MIN_BBOX_PX or h < MIN_BBOX_PX:
        return None
    if w > MAX_BBOX_PX or h > MAX_BBOX_PX:
        return None

    return x1, y1, w, h


# =============================================
# 실시간 디버그 화면 렌더링
# =============================================
def draw_debug_overlay(img_bgr, detections, frame_idx, saved_car, saved_truck):
    """
    CARLA 카메라 프레임 위에 Ground Truth BBox + 라벨 오버레이

    Args:
        img_bgr   : 원본 BGR 이미지
        detections: [(label, bbox, vehicle_id), ...] 리스트
        frame_idx : 현재 프레임 번호
        saved_car, saved_truck: 누적 저장 수

    Returns:
        display: 오버레이가 그려진 이미지
    """
    display = img_bgr.copy()

    for label, bbox, vid in detections:
        x, y, w, h = bbox
        color = COLOR_CAR if label == "car" else COLOR_TRUCK

        # BBox 사각형
        cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)

        # 라벨 배경
        text = f"{label} #{vid}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(display, (x, y - th - 6), (x + tw + 4, y), color, -1)
        cv2.putText(display, text, (x + 2, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # 상단 정보 패널
    panel_h = 60
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (display.shape[1], panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

    info = (f"Frame: {frame_idx}  |  "
            f"Detected: {len(detections)}  |  "
            f"Saved - Car: {saved_car}  Truck: {saved_truck}  |  "
            f"NPC: {NUM_NPC}  |  [Q] Quit")
    cv2.putText(display, info, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    return display


# =============================================
# 메인 데이터 수집 루프
# =============================================
def collect_dataset(world, camera, img_queue, vehicles):
    """
    Ground Truth 기반 차량 이미지 자동 캡처 + 화면 디버그 뷰

    변경사항:
        - is_valid_vehicle() 체크 추가 → 이륜차/trailer 건너뜀
        - MAX_BBOX_PX 확대 → 트럭 전체가 한 장에 저장
    """
    K = build_projection_matrix(IMG_WIDTH, IMG_HEIGHT, FOV)

    # 차량별 마지막 캡처 시각 (중복 방지)
    last_capture: dict[int, float] = {}

    saved_car   = 0
    saved_truck = 0
    frame_idx   = 0

    cv2.namedWindow("CARLA DataCollector - CCTV View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("CARLA DataCollector - CCTV View", 1280, 720)

    print(f"[DataCollector] 데이터 수집 시작 ({SIM_DURATION}초) — [Q]키로 종료")
    start_time = time.time()

    try:
        while time.time() - start_time < SIM_DURATION:
            world.tick()

            # 이미지 큐에서 최신 프레임 가져오기
            try:
                carla_image = img_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            # CARLA BGRA → OpenCV BGR 변환
            raw = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
            raw = raw.reshape((IMG_HEIGHT, IMG_WIDTH, 4))
            frame_bgr = raw[:, :, :3].copy()

            # 원본 저장 (선택)
            if SAVE_RAW:
                raw_path = os.path.join(RAW_DIR, f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(raw_path, frame_bgr)

            # 현재 월드의 모든 차량 가져오기
            actor_list = world.get_actors().filter("vehicle.*")
            now = time.time()

            detections = []

            cam_loc = camera.get_location()   # 가시성 체크용 카메라 위치

            for vehicle in actor_list:
                vid   = vehicle.id
                bp_id = vehicle.type_id

                # ── 이륜차 / trailer 제외 (나무처럼 보이는 샘플 & 트럭 분할 방지) ──
                if not is_valid_vehicle(bp_id):
                    continue

                # ── 벽/건물에 가려진 차량 제외 (cast_ray 가시성 체크) ──
                if not is_vehicle_visible(world, cam_loc, vehicle):
                    continue

                label = get_vehicle_label(bp_id)

                # 3D BBox → 2D 픽셀 BBox
                bbox = get_vehicle_bbox_pixels(vehicle, camera, K,
                                               IMG_WIDTH, IMG_HEIGHT)
                if bbox is None:
                    continue

                detections.append((label, bbox, vid))

                # 쿨다운 체크 (같은 차량 중복 캡처 방지)
                if now - last_capture.get(vid, 0) < CAPTURE_COOLDOWN:
                    continue

                # 크롭 & 저장
                x, y, w, h = bbox
                crop = frame_bgr[y:y + h, x:x + w]
                if crop.size == 0:
                    continue

                ts = int(now * 1000)
                filename = f"{label}_{vid}_{ts}.jpg"
                save_path = os.path.join(OUTPUT_DIR, label, filename)
                cv2.imwrite(save_path, crop)
                last_capture[vid] = now

                if label == "car":
                    saved_car += 1
                else:
                    saved_truck += 1

            # 디버그 오버레이 화면 표시
            display = draw_debug_overlay(frame_bgr, detections,
                                         frame_idx, saved_car, saved_truck)
            cv2.imshow("CARLA DataCollector - CCTV View", display)

            frame_idx += 1

            # Q 키 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[DataCollector] 사용자 종료 요청")
                break

            # 진행 상황 5초마다 출력
            if frame_idx % 50 == 0:
                elapsed = time.time() - start_time
                print(f"[DataCollector] {elapsed:.0f}s | "
                      f"Frame {frame_idx} | "
                      f"Car: {saved_car}  Truck: {saved_truck}")

    finally:
        cv2.destroyAllWindows()

    print(f"\n[DataCollector] 수집 완료!")
    print(f"  총 프레임  : {frame_idx}")
    print(f"  저장 car   : {saved_car}장")
    print(f"  저장 truck : {saved_truck}장")
    print(f"  저장 경로  : {OUTPUT_DIR}/")


# =============================================
# 진입점
# =============================================
def main():
    setup_output_dirs()
    client, world = connect_to_carla()

    # 동기 모드 설정 (tick() 제어를 위해 권장)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05   # 20 FPS
    world.apply_settings(settings)

    intersection_loc = find_intersection(world)
    camera, img_queue = spawn_cctv_camera(world, intersection_loc)

    # 날씨 랜덤 선택 (다양한 데이터 확보)
    weather = random.choice(["clear", "clear", "cloudy", "rain"])
    set_weather(world, weather)

    vehicles = spawn_npc_vehicles(client, world, num=NUM_NPC)

    # 배경 워밍업 (차량들이 이동을 시작하도록 대기)
    print("[DataCollector] 배경 워밍업 중...")
    for _ in range(50):
        world.tick()
        try:
            img_queue.get_nowait()
        except queue.Empty:
            pass

    try:
        collect_dataset(world, camera, img_queue, vehicles)
    finally:
        # 동기 모드 해제
        settings.synchronous_mode = False
        world.apply_settings(settings)

        print("[DataCollector] 정리 중...")
        camera.stop()
        camera.destroy()
        for v in vehicles:
            v.destroy()
        print("[DataCollector] 종료 완료")


if __name__ == "__main__":
    main()
