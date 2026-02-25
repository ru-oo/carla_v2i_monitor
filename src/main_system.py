"""
main_system.py
==============
담당: 전체 팀 (최종 통합 실행 파일)
브랜치: main (PR을 통해 통합)

역할:
    - data_collector.py, vision_processor.py, classifier.py를 연결하는 메인 루프
    - CARLA 프레임 수신 → 비전 처리 → 차종 분류 → 3D 지도 시각화
    - 최종 관제 화면(3D 레이더 맵 + 차량 정보 오버레이) 렌더링

실행 방법:
    python src/main_system.py
"""

import cv2
import numpy as np
import carla
import time

from vision_processor import VisionProcessor, DetectedObject
from classifier import VehicleClassifier

# =============================================
# 설정값
# =============================================
CARLA_HOST = "localhost"
CARLA_PORT = 2000
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
RADAR_MAP_SIZE = 400       # 3D 레이더 맵 크기 (픽셀)

# 차종별 시각화 색상 (BGR)
COLORS = {
    "car": (0, 255, 0),     # 초록
    "truck": (0, 100, 255), # 주황
    "unknown": (200, 200, 200),
}


# =============================================
# 레이더 맵 렌더러
# =============================================
class RadarMapRenderer:
    """
    Top-view 3D 레이더 맵 렌더링 클래스
    감지된 차량의 물리적 위치를 2D 맵에 점으로 표시합니다.
    """

    def __init__(self, map_size: int = RADAR_MAP_SIZE):
        self.map_size = map_size
        self.scale = map_size / 40.0   # 40m x 40m 교차로를 맵에 매핑

    def render(self, objects: list) -> np.ndarray:
        """
        감지 객체 목록을 받아 레이더 맵 이미지 반환

        Args:
            objects: (DetectedObject, vehicle_type) 튜플 리스트

        Returns:
            radar_map: BGR 레이더 맵 이미지
        """
        radar = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)

        # 도로/교차로 배경 그리기
        self._draw_intersection(radar)

        # 차량 위치 표시
        for obj, vehicle_type in objects:
            px = int(obj.world_x * self.scale)
            py = int(obj.world_y * self.scale)

            if 0 <= px < self.map_size and 0 <= py < self.map_size:
                color = COLORS.get(vehicle_type, COLORS["unknown"])
                cv2.circle(radar, (px, py), 8, color, -1)
                cv2.putText(
                    radar, vehicle_type[:1].upper(),
                    (px + 10, py + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
                )

        # 범례
        self._draw_legend(radar)
        return radar

    def _draw_intersection(self, canvas: np.ndarray):
        """교차로 배경 격자 그리기"""
        # 도로 (어두운 회색)
        road_color = (60, 60, 60)
        road_w = int(self.map_size * 0.25)
        center = self.map_size // 2

        cv2.rectangle(canvas,
                      (center - road_w // 2, 0),
                      (center + road_w // 2, self.map_size),
                      road_color, -1)
        cv2.rectangle(canvas,
                      (0, center - road_w // 2),
                      (self.map_size, center + road_w // 2),
                      road_color, -1)

        # 중앙선 (노란색 점선)
        for i in range(0, self.map_size, 20):
            cv2.line(canvas, (center, i), (center, i + 10), (0, 200, 200), 1)
            cv2.line(canvas, (i, center), (i + 10, center), (0, 200, 200), 1)

    def _draw_legend(self, canvas: np.ndarray):
        """범례 표시"""
        cv2.circle(canvas, (15, 15), 6, COLORS["car"], -1)
        cv2.putText(canvas, "Car", (25, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS["car"], 1)
        cv2.circle(canvas, (15, 35), 6, COLORS["truck"], -1)
        cv2.putText(canvas, "Truck", (25, 39),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS["truck"], 1)


# =============================================
# 메인 관제 시스템
# =============================================
class V2IMonitorSystem:
    """
    V2I 스마트 교차로 3D 관제 시스템 메인 클래스

    파이프라인:
        CARLA Frame → VisionProcessor → VehicleClassifier → RadarMapRenderer → 화면 출력
    """

    def __init__(self):
        self.vision = VisionProcessor()
        self.classifier = VehicleClassifier()
        self.radar = RadarMapRenderer()
        print("[V2IMonitor] 시스템 초기화 완료")

    def process_frame(self, frame: np.ndarray):
        """
        단일 프레임 처리 및 시각화

        Args:
            frame: CARLA 카메라 BGR 프레임

        Returns:
            display_frame: 오버레이가 추가된 최종 출력 프레임
        """
        # 1. 비전 처리: 움직이는 차량 감지
        detected_objects = self.vision.process_frame(frame)

        # 2. 차종 분류: 각 감지 객체를 CNN으로 분류
        classified = []
        for obj in detected_objects:
            x, y, w, h = obj.bbox
            crop = frame[y:y + h, x:x + w]
            vehicle_type = self.classifier.classify(crop)
            classified.append((obj, vehicle_type))

        # 3. 원본 프레임에 오버레이
        display = self._overlay_detection(frame.copy(), classified)

        # 4. 레이더 맵 생성 및 합성
        radar_map = self.radar.render(classified)
        display = self._compose_display(display, radar_map)

        return display

    def _overlay_detection(self, frame: np.ndarray, classified: list) -> np.ndarray:
        """감지 결과를 원본 프레임에 오버레이"""
        for obj, vehicle_type in classified:
            x, y, w, h = obj.bbox
            color = COLORS.get(vehicle_type, COLORS["unknown"])

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = f"{vehicle_type} ({obj.world_x:.1f}m, {obj.world_y:.1f}m)"
            cv2.putText(frame, label, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 감지 수 표시
        count_text = f"Detected: {len(classified)}"
        cv2.putText(frame, count_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        return frame

    def _compose_display(self, main_frame: np.ndarray, radar_map: np.ndarray) -> np.ndarray:
        """메인 화면 우측 하단에 레이더 맵 합성"""
        h, w = main_frame.shape[:2]
        map_h, map_w = radar_map.shape[:2]

        margin = 10
        y_start = h - map_h - margin
        x_start = w - map_w - margin

        # 반투명 배경
        overlay = main_frame.copy()
        cv2.rectangle(overlay,
                      (x_start - 5, y_start - 5),
                      (x_start + map_w + 5, y_start + map_h + 5),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, main_frame, 0.3, 0, main_frame)

        main_frame[y_start:y_start + map_h, x_start:x_start + map_w] = radar_map
        return main_frame


def main():
    """CARLA 연결 및 메인 관제 루프 실행"""
    system = V2IMonitorSystem()

    # CARLA 연결
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(10.0)
    world = client.get_world()

    # 카메라 설치 (data_collector.py의 설정과 동일하게 맞춰야 함)
    blueprint_library = world.get_blueprint_library()
    camera_bp = blueprint_library.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(WINDOW_WIDTH))
    camera_bp.set_attribute("image_size_y", str(WINDOW_HEIGHT))

    camera_transform = carla.Transform(
        carla.Location(x=0.0, y=0.0, z=20.0),
        carla.Rotation(pitch=-60, yaw=0, roll=0),
    )
    camera = world.spawn_actor(camera_bp, camera_transform)

    frame_buffer = []

    def on_image(image):
        img_array = np.frombuffer(image.raw_data, dtype=np.uint8)
        img_array = img_array.reshape((image.height, image.width, 4))
        frame_buffer.append(img_array[:, :, :3].copy())

    camera.listen(on_image)

    print("[V2IMonitor] 관제 시스템 시작 (q: 종료)")
    cv2.namedWindow("V2I Smart Intersection Monitor", cv2.WINDOW_NORMAL)

    try:
        while True:
            world.tick()

            if frame_buffer:
                frame = frame_buffer.pop()
                display = system.process_frame(frame)
                cv2.imshow("V2I Smart Intersection Monitor", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.destroy()
        cv2.destroyAllWindows()
        print("[V2IMonitor] 종료 완료")


if __name__ == "__main__":
    main()
