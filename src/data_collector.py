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
    - data/vehicle_images/van/   ← 밴 크롭 이미지
    - data/vehicle_images/bus/   ← 버스 크롭 이미지
    - data/raw/                  ← 원본 프레임 저장 (선택)

실행 방법:
    1. CARLA 서버 먼저 실행: ./CarlaUE4.sh  (또는 CarlaUE4.exe)
    2. python src/data_collector.py
"""

# test_git
import carla
import itertools
import random
import time
import os
import cv2
import numpy as np
import queue
import math
import traceback   # 예외 전체 스택 트레이스 출력 (원인 진단용)

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
MIN_BBOX_PX      = 20           # 30 → 20 (CCTV 뷰에서 멀리 보이는 차량도 수집)
MAX_BBOX_PX      = 1000          # 400 → 1000 (트럭/버스 전체가 한 장에 들어오도록)
CAPTURE_COOLDOWN = 0.5          # 중복 캡처 방지 쿨다운 (0.8 → 0.5초로 단축)
MIN_CAPTURE_DIST = 2.0          # m — 마지막 캡처 위치에서 이 거리 이상 이동해야 재캡처
                                #   (4.0 → 2.0m 완화, 정지 차량은 10초 후 예외 허용)
STATIONARY_RECAPTURE_SEC = 10.0 # 정지 차량도 이 시간(초) 지나면 위치 무관 재캡처 허용

# --- Stuck 차량 감지 설정 ---
STUCK_CHECK_INTERVAL = 100      # 프레임마다 stuck 체크
STUCK_SPEED_THRESHOLD = 0.5     # m/s 미만이면 정지로 판단
STUCK_MAX_COUNT  = 3            # 3회 연속(= 약 15초) 정지 감지 시 제거·재생성

# --- cast_ray 캐싱 설정 ---
VISIBILITY_CACHE_FRAMES = 10    # 10프레임(0.5초)마다 가시성 재계산

# --- 시뮬레이션 규모 ---
NUM_NPC       = 80              # 40 → 100 (차량 수 증가)
SIM_DURATION  = 300             # 180 → 300초 (더 많은 데이터 수집)

# --- 수집 대상 차종 (CARLA base_type 기준) ---
# car / truck / van / bus 4개 클래스를 각각 개별 폴더에 수집
TARGET_CLASSES   = ["car", "truck", "van", "bus"]
VALID_BASE_TYPES = set(TARGET_CLASSES)   # 수집 대상 base_type 집합

# 이륜차 base_type 값 (CARLA blueprint 속성) — 수집 완전 제외
EXCLUDE_BASE_TYPES = {"motorcycle", "bicycle"}

# 화면 표시용 색상 (BGR) — 클래스별
CLASS_COLORS = {
    "car":   (0, 255, 0),       # 초록
    "truck": (0, 100, 255),     # 주황
    "van":   (0, 220, 255),     # 노랑
    "bus":   (255, 0, 200),     # 보라
    "other": (200, 200, 200),
}

# --- 수집 목표량 (클래스당) ---
# 4개 클래스 균등 수집 (car / truck / van / bus 각 2,500장 = 총 10,000장)
TARGET_PER_CLASS = 2500    # 클래스당 수집 목표 장수

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

# --- 맵 로테이션 설정 ---
# CARLA 공식 지원 일반 도심 맵 8종 전체 순환
MAP_LIST = [
    "Town01", "Town02", "Town03", "Town04", 
    "Town05", "Town06", "Town07", "Town10HD"
]

FRAMES_PER_MAP = 3000  # 맵당 수집 프레임 수


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


def get_vehicle_class_bp(bp) -> str:
    """
    블루프린트 객체의 수집 클래스 반환 (스폰/필터링 단계용)

    Returns:
        TARGET_CLASSES 중 해당 클래스 이름, 또는 "" (수집 대상 아님)
    """
    base_type = get_base_type_from_bp(bp)
    return base_type if base_type in VALID_BASE_TYPES else ""


def get_vehicle_class_actor(vehicle) -> str:
    """
    액터 객체의 수집 클래스 반환 (collect_dataset 단계용)

    Returns:
        TARGET_CLASSES 중 해당 클래스 이름, 또는 "" (수집 대상 아님)
    """
    base_type = get_base_type_from_actor(vehicle)
    return base_type if base_type in VALID_BASE_TYPES else ""


def is_valid_vehicle_bp(bp) -> bool:
    """
    스폰 단계: 수집 대상 4륜 차량 블루프린트인지 확인

    수집 대상: car / truck / van / bus (VALID_BASE_TYPES)
    제외 대상: motorcycle / bicycle / trailer / 기타 base_type
    """
    if not get_vehicle_class_bp(bp):
        return False
    if "trailer" in bp.id.lower():
        return False
    return True


def is_valid_vehicle_actor(vehicle) -> bool:
    """
    수집 단계: 수집 대상 차종 액터인지 확인

    base_type 속성이 없는 구버전 CARLA 환경을 위한 fallback 포함
    """
    if not get_vehicle_class_actor(vehicle):
        return False
    if "trailer" in vehicle.type_id.lower():
        return False
    return True


def setup_output_dirs():
    """학습 데이터 저장 폴더 생성 (TARGET_CLASSES 기준)"""
    for cls in TARGET_CLASSES:
        os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)
    if SAVE_RAW:
        os.makedirs(RAW_DIR, exist_ok=True)
    print(f"[DataCollector] 출력 폴더 준비 완료: {OUTPUT_DIR}/  "
          f"({' | '.join(TARGET_CLASSES)})")


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

    # junction 진입로 waypoint 중 가장 SW 방향 선택 (waypoint 객체 보존)
    best_wp_obj = None
    best_score  = -float("inf")
    for entry_wp, _ in junction.get_waypoints(carla.LaneType.Driving):
        loc = entry_wp.transform.location
        # SW 점수: 서쪽(x 감소) + 남쪽(y 증가) 합
        score = (junction_center.x - loc.x) + (loc.y - junction_center.y)
        if score > best_score:
            best_score  = score
            best_wp_obj = entry_wp

    if best_wp_obj is None:
        best_loc = junction_center
    else:
        # ── 핵심 수정 ──
        # junction.get_waypoints()가 반환하는 wp는 교차로 경계에 있어
        # 교차로 중심과의 수평 거리가 수 m에 불과한 경우가 많음.
        # → 이 상태로 카메라를 6m 높이에 올리면 pitch가 거의 수직(바닥 응시) 됨.
        # previous(MOUNT_BACK_DIST)로 도로를 따라 교차로에서 멀어지는 방향으로 이동,
        # 카메라가 교차로를 적절한 각도로 내려다볼 수 있는 거리를 확보.
        MOUNT_BACK_DIST = 15.0   # 교차로 경계에서 도로 따라 후퇴할 거리 (m)
        prev_wps = best_wp_obj.previous(MOUNT_BACK_DIST)
        best_loc = (prev_wps[0].transform.location if prev_wps
                    else best_wp_obj.transform.location)

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

    # ── 수평 거리 최솟값 보장 ──
    # horiz_dist < MIN_HORIZ_DIST 이면 pitch가 너무 가팔라져 바닥만 찍힘.
    # 교차로 반대 방향으로 카메라를 강제 이동하여 최소 수평 거리 확보.
    MIN_HORIZ_DIST = 12.0
    if horiz_dist < MIN_HORIZ_DIST:
        if horiz_dist > 0.1:
            ratio = MIN_HORIZ_DIST / horiz_dist
            cam_x = junction_center.x - dx * ratio
            cam_y = junction_center.y - dy * ratio
        else:
            # 수평 벡터가 거의 0 → 임의 방향(SW)으로 강제 배치
            cam_x = junction_center.x - MIN_HORIZ_DIST * 0.7
            cam_y = junction_center.y + MIN_HORIZ_DIST * 0.7
        dx = junction_center.x - cam_x
        dy = junction_center.y - cam_y
        horiz_dist = math.sqrt(dx * dx + dy * dy)
        print(f"[DataCollector] ⚠ 수평 거리 부족 → 카메라 강제 이동: "
              f"({cam_x:.1f}, {cam_y:.1f})  horiz={horiz_dist:.1f}m")

    yaw   = math.degrees(math.atan2(dy, dx))
    pitch = math.degrees(math.atan2(dz, horiz_dist))
    # pitch 클램핑: -50° 이상 유지 (수직 바닥 응시 방지 최후 안전망)
    pitch = max(pitch, -50.0)

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

    4개 클래스(car / truck / van / bus) 균등 순환 스폰:
        - TARGET_CLASSES 순서로 round-robin 순환하여 25:25:25:25 비율 목표
        - 클래스 내 블루프린트를 itertools.cycle로 순환
          → 동일 모델 연속 스폰 방지 + 모든 차량 모델 균등 수집
    """
    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    # 클래스별 블루프린트 분류 및 셔플 (수집 다양성 확보)
    class_bps: dict[str, list] = {cls: [] for cls in TARGET_CLASSES}
    for bp in bp_lib.filter("vehicle.*"):
        if not is_valid_vehicle_bp(bp):
            continue
        cls = get_vehicle_class_bp(bp)
        if cls in class_bps:
            class_bps[cls].append(bp)

    for bps in class_bps.values():
        random.shuffle(bps)     # 스폰 순서 무작위화

    # 모든 모델이 균등 선택되도록 cycle 이터레이터 생성
    # → 같은 모델이 연속으로 반복되지 않고 블루프린트 전체를 순환
    class_iters: dict[str, object] = {
        cls: itertools.cycle(bps) if bps else None
        for cls, bps in class_bps.items()
    }

    bp_summary = "  ".join(f"{cls}: {len(class_bps[cls])}종"
                           for cls in TARGET_CLASSES)
    print(f"[DataCollector] 유효 블루프린트 — {bp_summary}  "
          f"(motorcycle/bicycle/trailer 제외)")

    vehicles = []
    tm = client.get_trafficmanager(8000)
    # 전역 차간 거리 2.5m — 너무 좁으면 추돌·정체 연쇄 발생
    tm.set_global_distance_to_leading_vehicle(2.5)

    # 4개 클래스를 순서대로 순환 스폰 (car→truck→van→bus→car→...)
    class_cycle = itertools.cycle(TARGET_CLASSES)
    for sp in spawn_points[:num]:
        target_cls = next(class_cycle)
        it = class_iters.get(target_cls)

        if it is None:
            # 해당 클래스 블루프린트 없으면 가용 클래스 중 랜덤 선택
            available = [c for c in TARGET_CLASSES if class_iters[c] is not None]
            if not available:
                continue
            it = class_iters[random.choice(available)]

        bp = next(it)
        actor = world.try_spawn_actor(bp, sp)
        if actor:
            actor.set_autopilot(True, tm.get_port())
            # ── TM 파라미터: 도로 준수 우선, 최소한의 다양성만 부여 ──
            # ignore_lights / 공격적 차선변경은 맵에 따라 차들을 보도·역주행으로 유도하므로 제거
            tm.auto_lane_change(actor, False)              # 자동 차선변경 비활성 (역주행 방지)
            tm.distance_to_leading_vehicle(actor, random.uniform(2.0, 4.0))
            tm.vehicle_percentage_speed_difference(actor, random.uniform(-10, 10))
            tm.ignore_lights_percentage(actor, 0)          # 신호등 완전 준수 (역주행·교차로 충돌 방지)
            tm.random_left_lanechange_percentage(actor, 5) # 5%만 허용 (거의 차선변경 안 함)
            tm.random_right_lanechange_percentage(actor, 5)
            vehicles.append(actor)

    # 실제 스폰된 클래스별 카운트 집계
    final_counts: dict[str, int] = {cls: 0 for cls in TARGET_CLASSES}
    for v in vehicles:
        cls = get_vehicle_class_actor(v)
        if cls in final_counts:
            final_counts[cls] += 1

    count_summary = "  ".join(f"{cls}: {final_counts[cls]}대"
                              for cls in TARGET_CLASSES)
    print(f"[DataCollector] NPC {len(vehicles)}대 생성 (목표 {num}대) — {count_summary}")
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
    차량의 3D Bounding Box 8개 꼭짓점을 픽셀에 투영하여 2D BBox 반환
    (화면 밖으로 삐져나간 대형 차량/버스의 꼭짓점 클리핑 문제 해결 버전)
    """
    bbox_3d  = vehicle.bounding_box
    transform = vehicle.get_transform()
    verts = bbox_3d.get_world_vertices(transform)

    us = []
    vs = []
    
    for v in verts:
        px = world_to_pixel(v, camera_actor, K)
        # 카메라 렌즈 앞(z > 0)에 있는 꼭짓점만 좌표 수집 (화면 밖이어도 수집)
        if px is not None:
            us.append(px[0])
            vs.append(px[1])

    # 카메라 앞에 꼭짓점이 2개 미만이면 박스를 그릴 수 없음
    if len(us) < 2:
        return None

    # 화면 밖 좌표를 포함한 전체 최소/최대 픽셀 좌표 계산
    x1, y1 = min(us), min(vs)
    x2, y2 = max(us), max(vs)

    # 이미지 해상도 경계선에 맞춰 BBox를 자름 (Clamp)
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w - 1))
    y2 = max(0, min(y2, img_h - 1))

    w = x2 - x1
    h = y2 - y1

    # 너무 작거나 큰 박스 제외
    if w < MIN_BBOX_PX or h < MIN_BBOX_PX:
        return None
    if w > MAX_BBOX_PX or h > MAX_BBOX_PX:
        return None

    return int(x1), int(y1), int(w), int(h)


# =============================================
# 실시간 디버그 화면 렌더링
# =============================================
def draw_debug_overlay(img_bgr, detections, frame_idx,
                       saved_counts: dict, weather_name: str = "", map_name: str = ""):
    """
    CARLA 카메라 프레임 위에 Ground Truth BBox + 라벨 + 수집 진행률 오버레이

    Args:
        img_bgr       : 원본 BGR 이미지
        detections    : [(label, bbox, vehicle_id), ...] 리스트
        frame_idx     : 현재 프레임 번호
        saved_counts  : {"car": n, "truck": n, "van": n, "bus": n} 누적 저장 수
        weather_name  : 현재 날씨 프리셋 이름 (표시용)

    Returns:
        display: 오버레이가 그려진 이미지
    """
    display = img_bgr.copy()

    for label, bbox, vid in detections:
        x, y, w, h = bbox
        color = CLASS_COLORS.get(label, CLASS_COLORS["other"])

        # BBox 사각형
        cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)

        # 라벨 배경
        text = f"{label} #{vid}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(display, (x, y - th - 6), (x + tw + 4, y), color, -1)
        cv2.putText(display, text, (x + 2, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # ── 상단 정보 패널 (1920×1080 기준: 4개 프로그레스 바 수용 220px) ──
    panel_h = 220
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (display.shape[1], panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

    # 기본 정보 텍스트 (맵 이름 + 날씨 포함)
    weather_label = weather_name if weather_name else "-"
    map_label = map_name if map_name else "-"
    info = (f"[{map_label}]  Frame: {frame_idx}  |  Detected: {len(detections)}  |  "
            f"Weather: {weather_label}  |  NPC: {NUM_NPC}  |  [Q] Quit")
    cv2.putText(display, info, (20, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

    # ── 수집 진행률 프로그레스 바 (4개 클래스, 1920px 기준) ──
    bar_w  = 500          # 프로그레스 바 최대 너비 (1920 기준 확대)
    bar_h  = 28           # 프로그레스 바 높이
    bar_x  = 20           # 시작 X
    gap    = 40           # 바 간격 (bar_h + 여백)
    bar_y0 = 68           # 첫 번째 바 Y 시작

    for i, cls in enumerate(TARGET_CLASSES):
        count = saved_counts.get(cls, 0)
        ratio = min(count / TARGET_PER_CLASS, 1.0)
        color = CLASS_COLORS.get(cls, CLASS_COLORS["other"])
        bar_y = bar_y0 + i * gap

        # 배경 바
        cv2.rectangle(display, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
        # 진행 바
        cv2.rectangle(display, (bar_x, bar_y),
                      (bar_x + int(bar_w * ratio), bar_y + bar_h), color, -1)
        # 텍스트
        label_str = f"{cls.capitalize():<5}: {count:>4}/{TARGET_PER_CLASS} ({ratio*100:5.1f}%)"
        cv2.putText(display, label_str,
                    (bar_x + bar_w + 16, bar_y + bar_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2)

    return display


# =============================================
# 메인 데이터 수집 루프
# =============================================
def collect_dataset(world, camera, img_queue, saved_counts: dict, map_name: str,
                    client=None, vehicles: list = None):
    """
    Ground Truth 기반 차량 이미지 자동 캡처 + 화면 디버그 뷰

    수정사항:
        - base_type 속성 기반 이륜차 필터링으로 교체
          (키워드 방식 누락 문제 완전 해결)
        - 클래스별 수집 캡(TARGET_PER_CLASS) 적용
          → car / truck / van / bus 각각 목표 수량 도달 시 해당 클래스 캡처 중단
          → 전 클래스 목표 달성 시 시뮬레이션 자동 종료
        - 날씨 로테이션 추가 (WEATHER_CHANGE_INTERVAL 프레임마다 순환)
          → WEATHER_PRESETS 7가지를 순서대로 적용
          → 단일 환경 과적합 방지 및 데이터 다양성 확보
        - FRAMES_PER_MAP 기반 종료 (SIM_DURATION 시간 방식 → 프레임 수 방식으로 교체)
        - NPC 차량 100프레임마다 생존 확인 → 절반 이하 소멸 시 자동 재생성
    """
    K = build_projection_matrix(IMG_WIDTH, IMG_HEIGHT, FOV)

    # 차량별 마지막 캡처 시각 (중복 방지)
    last_capture: dict[int, float] = {}
    # 차량별 마지막 캡처 위치 (x, y) — 신호 대기·정체 중 같은 장면 반복 캡처 방지
    last_capture_pos: dict[int, tuple] = {}

    # ── cast_ray 가시성 캐시: vid -> (last_frame, visible) ──
    visibility_cache: dict[int, tuple] = {}

    # ── Stuck 차량 카운터: vid -> 연속 정지 횟수 ──
    stuck_counts: dict[int, int] = {}

    # ── actor_list 캐시 (3프레임마다 갱신) ──
    # get_actors()는 CARLA 서버 RPC 호출로 비용이 있음 → 매 프레임 호출 대신 캐싱
    actor_list_cache: list = []
    ACTOR_CACHE_INTERVAL = 3

    # ★ saved_counts는 main()에서 넘어온 global_saved_counts 참조 —
    #   절대 새 dict로 재초기화하지 말 것 (맵 간 누적 카운트 파괴됨)
    frame_idx = 0
    user_quit = False   # Q키 종료 시 main() 루프도 함께 빠져나오기 위한 플래그

    # ── 날씨 로테이션 초기화 ──
    weather_idx     = 0
    current_weather = WEATHER_PRESETS[0]
    set_weather(world, current_weather)   # 첫 번째 날씨 즉시 적용

    cv2.namedWindow("CARLA DataCollector - CCTV View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("CARLA DataCollector - CCTV View", 1280, 720)

    progress_str = "  ".join(
        f"{cls}: {saved_counts.get(cls, 0)}/{TARGET_PER_CLASS}"
        for cls in TARGET_CLASSES
    )
    print(
        f"[DataCollector] 수집 시작 [{map_name}] ({FRAMES_PER_MAP}프레임) — [Q]키로 전체 종료\n"
        f"  현재 누적: {progress_str}\n"
        f"  날씨 로테이션: {len(WEATHER_PRESETS)}종 × {WEATHER_CHANGE_INTERVAL}프레임 주기"
    )
    start_time = time.time()

    consecutive_tick_errors = 0   # 연속 world.tick() 오류 카운터 (CARLA 연결 끊김 감지용)
    try:
        while frame_idx < FRAMES_PER_MAP:
            # ── world.tick() 오류 내성 ──
            # CARLA 일시적 오류 시 최대 5회 재시도, 연속 실패 시 이 맵 세션 강제 종료
            try:
                world.tick()
                consecutive_tick_errors = 0
            except Exception as _tick_err:
                consecutive_tick_errors += 1
                print(f"[DataCollector] ⚠ world.tick() 실패 "
                      f"({consecutive_tick_errors}/5): {_tick_err}")
                if consecutive_tick_errors >= 5:
                    print("[DataCollector] tick 연속 실패 → 이 맵 세션 강제 종료")
                    break
                time.sleep(0.5)
                continue

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

            # ── NPC 수 유지 + Stuck 차량 감지 (STUCK_CHECK_INTERVAL 프레임마다) ──
            if client is not None and vehicles is not None and frame_idx > 0 and frame_idx % STUCK_CHECK_INTERVAL == 0:
                live = [v for v in vehicles if v.is_alive]

                # 속도 체크 → STUCK_MAX_COUNT 연속 정지이면 제거
                to_remove = []
                for v in live:
                    try:
                        vel = v.get_velocity()
                        speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
                    except Exception:
                        continue
                    if speed < STUCK_SPEED_THRESHOLD:
                        stuck_counts[v.id] = stuck_counts.get(v.id, 0) + 1
                        if stuck_counts[v.id] >= STUCK_MAX_COUNT:
                            to_remove.append(v)
                            stuck_counts.pop(v.id, None)
                    else:
                        stuck_counts.pop(v.id, None)

                if to_remove:
                    print(f"[DataCollector] ⚠ stuck 차량 {len(to_remove)}대 제거 → 재생성")
                    for v in to_remove:
                        try:
                            if v.is_alive:
                                v.destroy()
                        except BaseException:
                            pass
                    live = [v for v in live if v.is_alive and v not in to_remove]

                need = NUM_NPC - len(live)
                if need > 0:
                    print(f"[DataCollector] ⚠ NPC {need}대 재생성 (생존: {len(live)}/{NUM_NPC})")
                    new_vehicles = spawn_npc_vehicles(client, world, num=need)
                    vehicles[:] = live + new_vehicles

            # 현재 월드의 모든 차량 가져오기 (3프레임마다 갱신)
            if frame_idx % ACTOR_CACHE_INTERVAL == 0:
                actor_list_cache = list(world.get_actors().filter("vehicle.*"))
            actor_list = actor_list_cache
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

                # ── 벽/건물에 가려진 차량 제외 (cast_ray — 캐시로 10프레임마다 재계산) ──
                cache_entry = visibility_cache.get(vid)
                if cache_entry and (frame_idx - cache_entry[0]) < VISIBILITY_CACHE_FRAMES:
                    visible = cache_entry[1]
                else:
                    visible = is_vehicle_visible(world, cam_loc, vehicle)
                    visibility_cache[vid] = (frame_idx, visible)
                if not visible:
                    continue

                label = get_vehicle_class_actor(vehicle)
                if not label:
                    continue

                # 3D BBox → 2D 픽셀 BBox
                bbox = get_vehicle_bbox_pixels(vehicle, camera, K,
                                               IMG_WIDTH, IMG_HEIGHT)
                if bbox is None:
                    continue

                detections.append((label, bbox, vid))

                # ── 클래스 캡(Cap) 체크 ──
                # 해당 클래스가 목표 수량에 도달하면 더 이상 저장하지 않음
                if saved_counts.get(label, 0) >= TARGET_PER_CLASS:
                    continue

                # 쿨다운 체크 (같은 차량 중복 캡처 방지)
                if now - last_capture.get(vid, 0) < CAPTURE_COOLDOWN:
                    continue

                # 이동 거리 체크 — 정지 차량은 STATIONARY_RECAPTURE_SEC 초 후 예외 허용
                cur_loc = vehicle.get_location()
                last_pos = last_capture_pos.get(vid)
                if last_pos is not None:
                    moved = math.sqrt(
                        (cur_loc.x - last_pos[0]) ** 2 +
                        (cur_loc.y - last_pos[1]) ** 2
                    )
                    time_since_capture = now - last_capture.get(vid, 0)
                    if moved < MIN_CAPTURE_DIST and time_since_capture < STATIONARY_RECAPTURE_SEC:
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
                last_capture_pos[vid] = (cur_loc.x, cur_loc.y)   # 캡처 위치 기록

                saved_counts[label] = saved_counts.get(label, 0) + 1

            # ── 전 클래스 목표 도달 시 자동 종료 ──
            if all(saved_counts.get(cls, 0) >= TARGET_PER_CLASS for cls in TARGET_CLASSES):
                count_str = "  ".join(f"{cls}: {saved_counts[cls]}장"
                                      for cls in TARGET_CLASSES)
                print(f"\n[DataCollector] ✓ 전 클래스 목표 달성! 자동 종료합니다.\n"
                      f"  {count_str}  (목표: {TARGET_PER_CLASS}장/클래스)")
                break

            # 디버그 오버레이 화면 표시 — 3프레임마다 1회 갱신 (imshow 병목 완화)
            if frame_idx % 3 == 0:
                try:
                    display = draw_debug_overlay(frame_bgr, detections,
                                                 frame_idx, saved_counts,
                                                 current_weather, map_name)
                    cv2.imshow("CARLA DataCollector - CCTV View", display)

                    # X 버튼으로 창이 닫혔는지 감지
                    if cv2.getWindowProperty("CARLA DataCollector - CCTV View",
                                             cv2.WND_PROP_VISIBLE) < 1:
                        print("[DataCollector] OpenCV 창 닫힘 감지 → 다음 맵으로 계속")
                        break
                except Exception:
                    pass   # 화면 표시 오류는 데이터 수집을 중단하지 않음

            frame_idx += 1

            # Q 키 누르면 이 맵 세션 + 전체 루프 모두 종료 (매 프레임 체크 유지)
            try:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[DataCollector] 사용자 종료 요청 — 전체 수집 중단")
                    user_quit = True
                    break
            except Exception:
                pass

            # 진행 상황 50프레임마다 출력
            if frame_idx % 50 == 0:
                elapsed = time.time() - start_time
                count_str = "  ".join(
                    f"{cls}: {saved_counts.get(cls, 0)}/{TARGET_PER_CLASS} "
                    f"({min(saved_counts.get(cls, 0) / TARGET_PER_CLASS * 100, 100):.1f}%)"
                    for cls in TARGET_CLASSES
                )
                print(f"[DataCollector] {elapsed:.0f}s | Frame {frame_idx} | "
                      f"Weather: {current_weather} | {count_str}")

    finally:
        # cv2.destroyAllWindows()가 Windows WM_QUIT 메시지를 처리하면서
        # 프로세스 종료 신호(SystemExit/BaseException)를 발생시킬 수 있으므로
        # except BaseException 으로 보호 (except Exception 은 SystemExit 를 잡지 못함)
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)   # 잔류 메시지 큐 비우기 (Windows 안정화)
        except BaseException:
            pass

    # ── 맵 세션 종료 메시지 (전체 수집 완료가 아닌 이 맵의 수집 종료) ──
    final_progress = "  ".join(
        f"{cls}: {saved_counts.get(cls, 0)}/{TARGET_PER_CLASS}"
        for cls in TARGET_CLASSES
    )
    print(f"\n[DataCollector] [{map_name}] 세션 완료 | 프레임: {frame_idx}")
    print(f"  누적 수집: {final_progress}")
    print(f"  저장 경로: {OUTPUT_DIR}/")
    return user_quit


# =============================================
# 진입점
# =============================================
def main():
    setup_output_dirs()
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    # Town10HD 등 대형 맵은 로드에 30~60초 소요 — 타임아웃 부족 시 RuntimeError 크래시
    client.set_timeout(120.0)

    # 맵이 바뀌어도 유지되는 전역 카운트 변수 (collect_dataset에 참조로 전달)
    global_saved_counts = {cls: 0 for cls in TARGET_CLASSES}
    user_quit = False   # Q키로 전체 종료 여부

    for map_name in MAP_LIST:
        if user_quit:   # 이전 맵에서 Q 키 → 루프 즉시 탈출
            break

        print(f"\n[DataCollector] === 맵 로드 중: {map_name} ===")

        # ★ finally 에서 None 체크 후 정리하기 위해 미리 초기화
        camera   = None
        vehicles = []
        world    = None
        tm       = None
        settings = None

        try:
            # ── 맵 로드 + CARLA 서버 안정화 ──
            # load_world 는 동기 블로킹 호출이지만, 반환 직후 서버 내부 초기화가
            # 아직 진행 중일 수 있으므로 sleep 으로 완전 안정화를 보장
            world = client.load_world(map_name)
            time.sleep(5.0)   # 맵 전환 후 CARLA 서버 완전 안정화 대기 (2→5초)

            # 트래픽 매니저 초기화
            tm = client.get_trafficmanager(8000)

            # 동기 모드 설정 (tick() 제어를 위해 권장)
            settings = world.get_settings()
            settings.synchronous_mode   = True
            settings.fixed_delta_seconds = 0.05   # 20 FPS
            world.apply_settings(settings)
            tm.set_synchronous_mode(True)

            # ── TM 사전 워밍업 (차량 스폰 전) ──
            # 맵 전환 직후 TM이 새 맵의 도로 토폴로지를 완전히 인식하기 전에
            # 차량을 스폰하면 TM이 이전 맵 기준으로 경로를 계산해 보도·역주행 발생.
            # 동기 모드 tick을 30회 진행해 TM이 새 도로 정보를 로드하도록 보장.
            print("[DataCollector] TM 도로 토폴로지 초기화 중...")
            for _ in range(30):
                try:
                    world.tick()
                except Exception:
                    pass

            intersection_loc = find_intersection(world)
            camera, img_queue = spawn_cctv_camera(world, intersection_loc)

            # 날씨는 collect_dataset() 내부에서 WEATHER_PRESETS 순서대로 자동 로테이션
            vehicles = spawn_npc_vehicles(client, world, num=NUM_NPC)

            # 배경 워밍업 (차량들이 올바른 차선에 안착하도록 충분히 대기)
            print("[DataCollector] 배경 워밍업 중...")
            for _ in range(100):   # 80 → 100 tick (도로 정착 시간 확보)
                try:
                    world.tick()
                except Exception:
                    pass
                try:
                    img_queue.get_nowait()
                except queue.Empty:
                    pass

            user_quit = collect_dataset(
                world, camera, img_queue, global_saved_counts, map_name,
                client=client, vehicles=vehicles,
            )

        except BaseException as exc:
            # ── BaseException 으로 변경한 이유 ──
            # CARLA Python 바인딩(pybind11)이 던지는 일부 예외는
            # Python Exception 계층을 올바르게 상속하지 않을 수 있음.
            # 또한 SystemExit / KeyboardInterrupt 도 여기서 일괄 처리해
            # finally 블록이 항상 정리를 수행하도록 보장.
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                # 명시적 종료 신호 → 루프 탈출 플래그 세팅 후 정리 진행
                print(f"\n[DataCollector] 종료 신호({type(exc).__name__}) → 전체 수집 중단")
                user_quit = True
            else:
                # CARLA 오류·타임아웃·스폰 실패 등 → 전체 트레이스 출력 후 다음 맵으로
                print(f"[DataCollector] ⚠  {map_name} 오류 발생 → 다음 맵으로 계속")
                traceback.print_exc()   # ← 실제 오류 원인을 콘솔에 출력 (진단 필수)

        finally:
            print(f"[DataCollector] {map_name} 에셋 정리 중...")

            # 카메라 안전 삭제
            if camera is not None:
                try:
                    camera.stop()
                    camera.destroy()
                except BaseException:
                    pass

            # NPC 차량 안전 삭제 (이미 파괴된 차량 에러 무시)
            for v in vehicles:
                try:
                    if v.is_alive:
                        v.destroy()
                except BaseException:
                    pass

            # ★ world.apply_settings(sync=False) 호출 금지 ★
            # sync → async 전환 시 CARLA C++ 클라이언트가 내부적으로 exit() 를
            # 호출해 프로세스가 조용히 종료되는 현상이 확인됨.
            # 대신 world.tick() 으로 서버 상태를 flush 한 뒤
            # load_world(reset_settings=True, 기본값) 가 자동으로 sync 모드를 해제.
            if world is not None:
                for _ in range(3):
                    try:
                        world.tick()
                    except BaseException:
                        break

            # 다음 맵 load_world 전 CARLA 서버 정리 시간 확보
            try:
                time.sleep(2.0)
            except BaseException:
                pass

        # ── 맵 루프 탈출 조건 (finally 블록 밖에서 평가) ──
        # finally 안에 break 를 두면 예외가 마스킹되거나 의도치 않은 루프 탈출이 발생
        all_done = all(
            global_saved_counts.get(cls, 0) >= TARGET_PER_CLASS
            for cls in TARGET_CLASSES
        )
        if all_done:
            print("[DataCollector] ✓ 전 클래스 목표 달성! 전체 수집 종료.")
        if all_done or user_quit:
            break

    # ── 전체 수집 결과 최종 출력 ──
    print(f"\n[DataCollector] ══ 전체 수집 완료 ══")
    for cls in TARGET_CLASSES:
        n = global_saved_counts.get(cls, 0)
        pct = min(n / TARGET_PER_CLASS * 100, 100)
        print(f"  {cls:<5}: {n:>4}장 / {TARGET_PER_CLASS}장 ({pct:.1f}%)")
    print(f"  저장 경로 : {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
