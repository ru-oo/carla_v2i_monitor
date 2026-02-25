"""
data_collector.py
=================
담당: 김예진 (CARLA 시뮬레이션 & 데이터 엔지니어링)
브랜치: feature/carla-sim

역할:
    - CARLA 시뮬레이터에서 교차로 고정 카메라(CCTV) 배치
    - 날씨/교통량 시나리오 설정 및 시뮬레이션 구동
    - Ground Truth 기반 차량 이미지 자동 캡처 및 저장
    - CNN 학습용 데이터셋 생성

산출물:
    - data/vehicle_images/ 폴더 내 차종별 이미지 (car/, truck/)
"""

import carla
import random
import time
import os
import cv2
import numpy as np

# =============================================
# 설정값 (Config)
# =============================================
CARLA_HOST = "localhost"
CARLA_PORT = 2000
CAMERA_HEIGHT = 20.0       # CCTV 높이 (m)
CAPTURE_INTERVAL = 0.5     # 이미지 저장 간격 (초)
OUTPUT_DIR = "data/vehicle_images"


def setup_output_dirs():
    """학습 데이터 저장 폴더 생성"""
    os.makedirs(os.path.join(OUTPUT_DIR, "car"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "truck"), exist_ok=True)
    print(f"[DataCollector] 출력 폴더 준비 완료: {OUTPUT_DIR}")


def connect_to_carla():
    """CARLA 서버에 연결"""
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(10.0)
    world = client.get_world()
    print(f"[DataCollector] CARLA 연결 성공: {world.get_map().name}")
    return client, world


def spawn_infrastructure_camera(world, intersection_location):
    """
    교차로 신호등 위에 고정 CCTV 카메라 설치

    Args:
        world: CARLA world 객체
        intersection_location: 카메라를 설치할 교차로 위치 (carla.Location)

    Returns:
        camera_sensor: 설치된 카메라 센서 액터
    """
    blueprint_library = world.get_blueprint_library()
    camera_bp = blueprint_library.find("sensor.camera.rgb")

    # 카메라 해상도 설정
    camera_bp.set_attribute("image_size_x", "1280")
    camera_bp.set_attribute("image_size_y", "720")
    camera_bp.set_attribute("fov", "90")

    # 교차로 위 고정 위치에 카메라 배치
    camera_transform = carla.Transform(
        carla.Location(
            x=intersection_location.x,
            y=intersection_location.y,
            z=CAMERA_HEIGHT,
        ),
        carla.Rotation(pitch=-60, yaw=0, roll=0),  # 내려다보는 각도
    )

    camera_sensor = world.spawn_actor(camera_bp, camera_transform)
    print(f"[DataCollector] CCTV 카메라 설치 완료: 위치={camera_transform.location}")
    return camera_sensor


def set_weather(world, weather_preset="clear"):
    """
    날씨 시나리오 설정

    Args:
        weather_preset: "clear" | "rain" | "night"
    """
    presets = {
        "clear": carla.WeatherParameters.ClearNoon,
        "rain": carla.WeatherParameters.HardRainNoon,
        "night": carla.WeatherParameters.ClearNight,
    }
    world.set_weather(presets.get(weather_preset, carla.WeatherParameters.ClearNoon))
    print(f"[DataCollector] 날씨 설정: {weather_preset}")


def setup_traffic(world, num_vehicles=30):
    """
    Traffic Manager를 사용하여 NPC 차량 자동 생성

    Args:
        num_vehicles: 생성할 차량 수

    Returns:
        vehicle_list: 생성된 차량 액터 리스트
    """
    blueprint_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    vehicle_list = []
    for i, spawn_point in enumerate(spawn_points[:num_vehicles]):
        vehicle_bp = random.choice(blueprint_library.filter("vehicle.*"))
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
        if vehicle:
            vehicle.set_autopilot(True)
            vehicle_list.append(vehicle)

    print(f"[DataCollector] NPC 차량 {len(vehicle_list)}대 생성 완료")
    return vehicle_list


def collect_dataset(world, camera_sensor, duration_sec=120):
    """
    Ground Truth 기반 차량 이미지 자동 캡처 및 저장

    Args:
        world: CARLA world 객체
        camera_sensor: CCTV 카메라 센서
        duration_sec: 데이터 수집 시간 (초)
    """
    frame_count = 0
    image_buffer = []

    def on_image(image):
        image_buffer.append(image)

    camera_sensor.listen(on_image)

    print(f"[DataCollector] 데이터 수집 시작 ({duration_sec}초)")
    start_time = time.time()

    while time.time() - start_time < duration_sec:
        world.tick()

        if image_buffer:
            image = image_buffer.pop()
            img_array = np.frombuffer(image.raw_data, dtype=np.uint8)
            img_array = img_array.reshape((image.height, image.width, 4))
            img_bgr = img_array[:, :, :3]

            # TODO: Ground Truth 좌표 기반 차량 Crop 및 분류 저장 로직 구현
            # 힌트: world.get_actors().filter("vehicle.*") 로 모든 차량 좌표 획득
            # 힌트: camera_sensor.get_transform() 으로 픽셀 좌표 변환

            frame_count += 1

        time.sleep(CAPTURE_INTERVAL)

    camera_sensor.stop()
    print(f"[DataCollector] 데이터 수집 완료. 총 {frame_count} 프레임 처리")


def main():
    setup_output_dirs()
    client, world = connect_to_carla()

    # 교차로 위치 설정 (맵에 따라 조정 필요)
    intersection_loc = carla.Location(x=0.0, y=0.0, z=0.0)

    camera = spawn_infrastructure_camera(world, intersection_loc)
    set_weather(world, weather_preset="clear")
    vehicles = setup_traffic(world, num_vehicles=30)

    try:
        collect_dataset(world, camera, duration_sec=120)
    finally:
        print("[DataCollector] 정리 중...")
        camera.destroy()
        for v in vehicles:
            v.destroy()
        print("[DataCollector] 종료 완료")


if __name__ == "__main__":
    main()
