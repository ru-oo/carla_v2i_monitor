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

# 트럭 계열 base_type 값 (CARLA blueprint 속성)
TRUCK_BASE_TYPES = {"truck", "van", "bus"}

# 이륜차 base_type 값 (CARLA blueprint 속성) — 키워드 매칭보다 정확
EXCLUDE_BASE_TYPES = {"motorcycle", "bicycle"}

# --- 수집 목표량 (클래스당) ---
# 학습 데이터 균형을 위해 car / truck 동일 수량 수집
TARGET_PER_CLASS = 5000    # 클래스당 수집 목표 장수

# --- 날씨 로테이션 설정 ---
# 동일한 환경에서만 수집 시 모델이 특정 조명·노면에 과적합되는 문제 방지
# 7가지 날씨를 순서대로 순환하며 다양한 조건의 데이터 확보
WEATHER_PRESETS = [
    "clear_noon",      # 맑은 낮         — 기본 기준 환경
    "cloudy_noon",     # 흐린 낮          — 그림자 없는 평탄한 조명
    "wet_noon",        # 젖은 노면 낮     — 비 온 후 반사광 있음
    "soft_rain_noon",  # 약한 비 낮       — 빗방울 노이즈 소량
    "hard_rain_noon",  # 강한 비 낮       — 시인성 저하 극단 조건
    "clear_sunset",    # 석양             — 역광·긴 그림자
    "clear_night",     # 밤               — 조도 극히 낮은 야간
]
# 20fps 기준 400 프레임 ≈ 20초마다 날씨 변경
# 300초 수집 시 약 15회 변경 → 7가지 날씨를 2회 이상 순환
WEATHER_CHANGE_INTERVAL = 400  # 날씨 변경 주기 (프레임 단위)


# =============================================
# 유틸 함수
# =============================================
def get_base_type_from_bp(bp) -> str:
    """
    블루프린트 객체에서 CARLA base_type 속성 추출 (spawn 단계에서 사용)

    base_type 속성값 예시: 'car', 'truck', 'van', 'bus', 'motorcycle', 'bicycle'
    키워드 매칭 방식보다 CARLA 내부 분류를 직접 활용하므로 신규 블루프린트에도 안정적
    """
    try:
        return bp.get_attribute("base_type").as_str().lower()
    except Exception:
        return ""


def get_base_type_from_actor(vehicle) -> str:
    """
    액터 객체에서 base_type 속성 추출 (collect_dataset 단계에서 사용)

    vehicle.attributes 딕셔너리는 블루프린트와 동일한 속성을 포함
    """
    return vehicle.attributes.get("base_type", "").lower()


def is_truck_bp(bp) -> bool:
    """블루프린트 객체로 트럭/대형차 여부 판단 (스폰 단계용)"""
    return get_base_type_from_bp(bp) in TRUCK_BASE_TYPES


def is_valid_vehicle_bp(bp) -> bool:
    """
    스폰 단계: 데이터 수집에 적합한 4륜 차량 블루프린트인지 확인

    제외 대상:
        - motorcycle / bicycle: base_type 속성으로 정확하게 필터링
          (기존 키워드 방식은 CARLA 신규 블루프린트 누락 위험)
        - trailer: 트럭과 별개 액터로 생성되어 동일 트럭이 분할 저장되는 원인
    """
    base_type = get_base_type_from_bp(bp)
    if base_type in EXCLUDE_BASE_TYPES:
        return False
    if "trailer" in bp.id.lower():
        return False
    return True


def is_truck_actor(vehicle) -> bool:
    """액터 객체로 트럭/대형차 여부 판단 (수집 단계용)"""
    return get_base_type_from_actor(vehicle) in TRUCK_BASE_TYPES


def is_valid_vehicle_actor(vehicle) -> bool:
    """
    수집 단계: 액터의 base_type 속성으로 이륜차 / trailer 제외

    base_type 속성이 없는 구버전 CARLA 환경을 위한 fallback 포함
    """
    base_type = get_base_type_from_actor(vehicle)
    if base_type in EXCLUDE_BASE_TYPES:
        return False
    if "trailer" in vehicle.type_id.lower():
        return False
    return True


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


def set_weather(world, preset: str = "clear_noon"):
    """
    날씨 설정 — WEATHER_PRESETS 리스트의 이름으로 호출

    지원 프리셋 (WEATHER_PRESETS 참고):
        clear_noon / cloudy_noon / wet_noon /
        soft_rain_noon / hard_rain_noon / clear_sunset / clear_night
    """
    params_map = {
        "clear_noon":      carla.WeatherParameters.ClearNoon,
        "cloudy_noon":     carla.WeatherParameters.CloudyNoon,
        "wet_noon":        carla.WeatherParameters.WetNoon,
        "soft_rain_noon":  carla.WeatherParameters.SoftRainNoon,
        "hard_rain_noon":  carla.WeatherParameters.HardRainNoon,
        "clear_sunset":    carla.WeatherParameters.ClearSunset,
        "clear_night":     carla.WeatherParameters.ClearNight,
    }
    world.set_weather(params_map.get(preset, carla.WeatherParameters.ClearNoon))
    print(f"[DataCollector] 날씨 변경 → {preset}")


def spawn_npc_vehicles(client, world, num=NUM_NPC):
    """
    NPC 차량 자동 생성 및 자율주행 활성화

    수정사항:
        - base_type 속성 기반 필터링으로 교체
          (키워드 방식은 신규 블루프린트 누락 위험 → base_type이 더 정확)
        - 승용차:트럭 = 5:5 (1:1) 균등 비율로 변경
          (7:3 비율은 데이터 불균형을 유발하여 모델 편향 원인)
    """
    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    # base_type 속성 기반 필터링 (motorcycle / bicycle / trailer 제외)
    valid_bps = [b for b in bp_lib.filter("vehicle.*") if is_valid_vehicle_bp(b)]
    truck_bps = [b for b in valid_bps if is_truck_bp(b)]
    car_bps   = [b for b in valid_bps if not is_truck_bp(b)]

    print(
        f"[DataCollector] 유효 블루프린트 — car: {len(car_bps)}종  "
        f"truck: {len(truck_bps)}종  "
        f"(motorcycle/bicycle/trailer 제외됨)"
    )

    vehicles = []
    tm = client.get_trafficmanager(8000)
    tm.set_global_distance_to_leading_vehicle(2.0)

    for sp in spawn_points[:num]:
        # 승용차:트럭 = 5:5 (1:1) — 균형 데이터셋 수집 목적
        if random.random() < 0.5 and truck_bps:
            bp = random.choice(truck_bps)
        elif car_bps:
            bp = random.choice(car_bps)
        else:
            bp = random.choice(valid_bps)

        actor = world.try_spawn_actor(bp, sp)
        if actor:
            actor.set_autopilot(True, tm.get_port())
            vehicles.append(actor)

    truck_count = sum(1 for v in vehicles if is_truck_actor(v))
    car_count   = len(vehicles) - truck_count
    print(
        f"[DataCollector] NPC 차량 {len(vehicles)}대 생성 (목표 {num}대) "
        f"— car: {car_count}대  truck: {truck_count}대"
    )
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
def draw_debug_overlay(img_bgr, detections, frame_idx,
                       saved_car, saved_truck, weather_name: str = ""):
    """
    CARLA 카메라 프레임 위에 Ground Truth BBox + 라벨 + 수집 진행률 오버레이

    Args:
        img_bgr      : 원본 BGR 이미지
        detections   : [(label, bbox, vehicle_id), ...] 리스트
        frame_idx    : 현재 프레임 번호
        saved_car    : 누적 car 저장 수
        saved_truck  : 누적 truck 저장 수
        weather_name : 현재 날씨 프리셋 이름 (표시용)

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

    # ── 상단 정보 패널 (높이 확장: 60 → 95) ──
    panel_h = 95
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (display.shape[1], panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

    # 기본 정보 텍스트 (날씨 이름 포함)
    weather_label = weather_name if weather_name else "-"
    info = (f"Frame: {frame_idx}  |  Detected: {len(detections)}  |  "
            f"Weather: {weather_label}  |  NPC: {NUM_NPC}  |  [Q] Quit")
    cv2.putText(display, info, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # ── 수집 진행률 프로그레스 바 ──
    bar_w     = 300          # 프로그레스 바 최대 너비
    bar_h     = 14           # 프로그레스 바 높이
    bar_x     = 10           # 시작 X
    bar_y_car   = 45         # Car 바 Y
    bar_y_truck = 68         # Truck 바 Y

    car_ratio   = min(saved_car   / TARGET_PER_CLASS, 1.0)
    truck_ratio = min(saved_truck / TARGET_PER_CLASS, 1.0)

    # Car 프로그레스 바
    cv2.rectangle(display, (bar_x, bar_y_car),
                  (bar_x + bar_w, bar_y_car + bar_h), (60, 60, 60), -1)
    cv2.rectangle(display, (bar_x, bar_y_car),
                  (bar_x + int(bar_w * car_ratio), bar_y_car + bar_h),
                  COLOR_CAR, -1)
    cv2.putText(display,
                f"Car  : {saved_car:>5}/{TARGET_PER_CLASS} ({car_ratio*100:5.1f}%)",
                (bar_x + bar_w + 10, bar_y_car + bar_h - 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_CAR, 1)

    # Truck 프로그레스 바
    cv2.rectangle(display, (bar_x, bar_y_truck),
                  (bar_x + bar_w, bar_y_truck + bar_h), (60, 60, 60), -1)
    cv2.rectangle(display, (bar_x, bar_y_truck),
                  (bar_x + int(bar_w * truck_ratio), bar_y_truck + bar_h),
                  COLOR_TRUCK, -1)
    cv2.putText(display,
                f"Truck: {saved_truck:>5}/{TARGET_PER_CLASS} ({truck_ratio*100:5.1f}%)",
                (bar_x + bar_w + 10, bar_y_truck + bar_h - 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_TRUCK, 1)

    return display


# =============================================
# 메인 데이터 수집 루프
# =============================================
def collect_dataset(world, camera, img_queue, vehicles):
    """
    Ground Truth 기반 차량 이미지 자동 캡처 + 화면 디버그 뷰

    수정사항:
        - base_type 속성 기반 이륜차 필터링으로 교체
          (키워드 방식 누락 문제 완전 해결)
        - 클래스별 수집 캡(TARGET_PER_CLASS) 적용
          → car / truck 각각 목표 수량 도달 시 해당 클래스 캡처 중단
          → 양쪽 모두 목표 달성 시 시뮬레이션 자동 종료
        - 날씨 로테이션 추가 (WEATHER_CHANGE_INTERVAL 프레임마다 순환)
          → WEATHER_PRESETS 7가지를 순서대로 적용
          → 단일 환경 과적합 방지 및 데이터 다양성 확보
    """
    K = build_projection_matrix(IMG_WIDTH, IMG_HEIGHT, FOV)

    # 차량별 마지막 캡처 시각 (중복 방지)
    last_capture: dict[int, float] = {}

    saved_car   = 0
    saved_truck = 0
    frame_idx   = 0

    # ── 날씨 로테이션 초기화 ──
    weather_idx     = 0
    current_weather = WEATHER_PRESETS[0]
    set_weather(world, current_weather)   # 첫 번째 날씨 즉시 적용

    cv2.namedWindow("CARLA DataCollector - CCTV View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("CARLA DataCollector - CCTV View", 1280, 720)

    print(
        f"[DataCollector] 데이터 수집 시작 ({SIM_DURATION}초) — [Q]키로 종료\n"
        f"  날씨 로테이션: {len(WEATHER_PRESETS)}종 × {WEATHER_CHANGE_INTERVAL}프레임 주기"
    )
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

            # ── 날씨 로테이션 체크 ──
            # frame_idx 기준으로 WEATHER_CHANGE_INTERVAL마다 다음 날씨로 순환
            new_weather_idx = (frame_idx // WEATHER_CHANGE_INTERVAL) % len(WEATHER_PRESETS)
            if new_weather_idx != weather_idx:
                weather_idx     = new_weather_idx
                current_weather = WEATHER_PRESETS[weather_idx]
                set_weather(world, current_weather)

            # 현재 월드의 모든 차량 가져오기
            actor_list = world.get_actors().filter("vehicle.*")
            now = time.time()

            detections = []

            cam_loc = camera.get_location()   # 가시성 체크용 카메라 위치

            for vehicle in actor_list:
                vid = vehicle.id

                # ── base_type 속성으로 이륜차 / trailer 제외 ──
                # 기존 키워드 방식은 CARLA 신규 블루프린트(예: 새 오토바이 모델)를
                # 누락할 수 있으므로 base_type 속성을 직접 사용
                if not is_valid_vehicle_actor(vehicle):
                    continue

                # ── 벽/건물에 가려진 차량 제외 (cast_ray 가시성 체크) ──
                if not is_vehicle_visible(world, cam_loc, vehicle):
                    continue

                label = "truck" if is_truck_actor(vehicle) else "car"

                # 3D BBox → 2D 픽셀 BBox
                bbox = get_vehicle_bbox_pixels(vehicle, camera, K,
                                               IMG_WIDTH, IMG_HEIGHT)
                if bbox is None:
                    continue

                detections.append((label, bbox, vid))

                # ── 클래스 캡(Cap) 체크 ──
                # 해당 클래스가 목표 수량에 도달하면 더 이상 저장하지 않음
                if label == "car"   and saved_car   >= TARGET_PER_CLASS:
                    continue
                if label == "truck" and saved_truck >= TARGET_PER_CLASS:
                    continue

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

            # ── 양쪽 클래스 모두 목표 도달 시 자동 종료 ──
            if saved_car >= TARGET_PER_CLASS and saved_truck >= TARGET_PER_CLASS:
                print(
                    f"\n[DataCollector] ✓ 목표 수집량 달성! 자동 종료합니다.\n"
                    f"  car: {saved_car}장  truck: {saved_truck}장 (목표: {TARGET_PER_CLASS}장)"
                )
                break

            # 디버그 오버레이 화면 표시
            display = draw_debug_overlay(frame_bgr, detections,
                                         frame_idx, saved_car, saved_truck,
                                         current_weather)
            cv2.imshow("CARLA DataCollector - CCTV View", display)

            frame_idx += 1

            # Q 키 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[DataCollector] 사용자 종료 요청")
                break

            # 진행 상황 5초마다 출력
            if frame_idx % 50 == 0:
                elapsed = time.time() - start_time
                car_pct   = min(saved_car   / TARGET_PER_CLASS * 100, 100)
                truck_pct = min(saved_truck / TARGET_PER_CLASS * 100, 100)
                print(
                    f"[DataCollector] {elapsed:.0f}s | Frame {frame_idx} | "
                    f"Weather: {current_weather} | "
                    f"Car: {saved_car}/{TARGET_PER_CLASS} ({car_pct:.1f}%)  "
                    f"Truck: {saved_truck}/{TARGET_PER_CLASS} ({truck_pct:.1f}%)"
                )

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

    # 날씨는 collect_dataset() 내부에서 WEATHER_PRESETS 순서대로 자동 로테이션
    # (기존 단일 랜덤 선택 제거 → 다양성 자동 확보)

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
