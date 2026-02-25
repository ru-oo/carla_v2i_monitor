"""
vision_processor.py
===================
담당: 김진수 (컴퓨터 비전 & OpenCV 알고리즘)
브랜치: feature/vision-logic

역할:
    - MOG2 배경 차분으로 움직이는 차량 마스크 추출
    - 형태학적 연산으로 노이즈 제거
    - IPM(Inverse Perspective Mapping)으로 2D → Top-view 3D 좌표 변환
    - 감지된 객체의 ROI(Region of Interest) 좌표 반환

산출물:
    - 감지된 차량의 (픽셀 좌표, 월드 좌표, Bounding Box) 리스트
"""
#김예진라면매운맛 ㅋㅋ
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


# =============================================
# 데이터 구조
# =============================================
@dataclass
class DetectedObject:
    """감지된 객체 정보"""
    pixel_x: int            # 원본 이미지 픽셀 X
    pixel_y: int            # 원본 이미지 픽셀 Y
    world_x: float          # Top-view 변환 후 물리적 X (m)
    world_y: float          # Top-view 변환 후 물리적 Y (m)
    bbox: Tuple[int, int, int, int]   # (x, y, w, h) Bounding Box
    area: float             # 마스크 내 픽셀 면적


# =============================================
# 설정값 (카메라 캘리브레이션에 맞게 수정 필요)
# =============================================

# IPM 변환용 소스 포인트 (카메라 시야 내 도로 4개 꼭짓점, 픽셀 단위)
# 실제 카메라 설치 후 캘리브레이션을 통해 측정해야 합니다.
IPM_SRC_POINTS = np.float32([
    [320, 400],   # 좌상단
    [960, 400],   # 우상단
    [1200, 700],  # 우하단
    [80, 700],    # 좌하단
])

# IPM 변환용 목적지 포인트 (Top-view 상의 실제 도로 크기, 미터 → 픽셀 변환)
# 예시: 20m x 20m 교차로를 400x400 픽셀로 매핑
IPM_DST_POINTS = np.float32([
    [0, 0],
    [400, 0],
    [400, 400],
    [0, 400],
])

PIXELS_PER_METER = 20.0    # IPM 출력에서 1m당 픽셀 수
MIN_CONTOUR_AREA = 500     # 최소 객체 면적 (노이즈 필터)
MAX_CONTOUR_AREA = 50000   # 최대 객체 면적 (이상치 필터)


class VisionProcessor:
    """
    V2I 비전 처리 메인 클래스
    MOG2 배경 차분 + IPM 좌표 변환을 담당합니다.
    """

    def __init__(self):
        # MOG2 배경 차분기 초기화
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=50,
            detectShadows=True,
        )

        # IPM 변환 행렬 사전 계산
        self.ipm_matrix = cv2.getPerspectiveTransform(IPM_SRC_POINTS, IPM_DST_POINTS)
        self.ipm_matrix_inv = np.linalg.inv(self.ipm_matrix)

        # 형태학적 연산 커널
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

        print("[VisionProcessor] 초기화 완료")

    def subtract_background(self, frame: np.ndarray) -> np.ndarray:
        """
        MOG2로 배경 차분 수행 → 움직이는 객체 마스크 반환

        Args:
            frame: 입력 BGR 프레임

        Returns:
            mask: 이진 마스크 (움직이는 영역=255, 정적 배경=0)
        """
        fg_mask = self.bg_subtractor.apply(frame)

        # 그림자(회색, 127) 제거 → 완전한 전경(흰색, 255)만 유지
        _, binary_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        return binary_mask

    def remove_noise(self, mask: np.ndarray) -> np.ndarray:
        """
        형태학적 연산으로 비/눈 노이즈 제거 및 객체 형태 정제

        Args:
            mask: 배경 차분 이진 마스크

        Returns:
            clean_mask: 노이즈 제거된 마스크
        """
        # Opening: 작은 노이즈 점 제거
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)
        # Closing: 객체 내 홀 메우기
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self.kernel_close)
        return closed

    def find_vehicle_contours(
        self, clean_mask: np.ndarray
    ) -> List[Tuple]:
        """
        마스크에서 차량 윤곽선 탐지 및 필터링

        Args:
            clean_mask: 노이즈 제거된 마스크

        Returns:
            valid_contours: 유효 면적 범위 내의 윤곽선 리스트
        """
        contours, _ = cv2.findContours(
            clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        valid_contours = [
            c for c in contours
            if MIN_CONTOUR_AREA < cv2.contourArea(c) < MAX_CONTOUR_AREA
        ]
        return valid_contours

    def pixel_to_world(self, pixel_x: int, pixel_y: int) -> Tuple[float, float]:
        """
        IPM 역투영으로 2D 픽셀 좌표 → Top-view 물리적 좌표 변환

        Args:
            pixel_x: 원본 이미지 X 픽셀
            pixel_y: 원본 이미지 Y 픽셀

        Returns:
            (world_x, world_y): 물리적 좌표 (미터 단위)
        """
        pixel_point = np.float32([[[pixel_x, pixel_y]]])
        topview_point = cv2.perspectiveTransform(pixel_point, self.ipm_matrix)

        world_x = float(topview_point[0][0][0]) / PIXELS_PER_METER
        world_y = float(topview_point[0][0][1]) / PIXELS_PER_METER

        return world_x, world_y

    def process_frame(self, frame: np.ndarray) -> List[DetectedObject]:
        """
        단일 프레임을 처리하여 감지된 객체 목록 반환
        (main_system.py에서 이 함수를 호출합니다)

        Args:
            frame: CARLA 카메라 BGR 프레임

        Returns:
            detected_objects: DetectedObject 리스트
        """
        mask = self.subtract_background(frame)
        clean_mask = self.remove_noise(mask)
        contours = self.find_vehicle_contours(clean_mask)

        detected_objects = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2

            world_x, world_y = self.pixel_to_world(center_x, center_y)

            obj = DetectedObject(
                pixel_x=center_x,
                pixel_y=center_y,
                world_x=world_x,
                world_y=world_y,
                bbox=(x, y, w, h),
                area=cv2.contourArea(contour),
            )
            detected_objects.append(obj)

        return detected_objects

    def get_topview_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        원본 프레임을 Top-view(조감도) 이미지로 변환 (시각화용)

        Args:
            frame: 원본 BGR 프레임

        Returns:
            topview: IPM 변환된 Top-view 이미지
        """
        topview = cv2.warpPerspective(frame, self.ipm_matrix, (400, 400))
        return topview


# =============================================
# 단독 실행 테스트 (개발 중 디버깅용)
# =============================================
def main():
    """웹캠 또는 영상 파일로 알고리즘 단독 테스트"""
    processor = VisionProcessor()

    # 테스트: 웹캠 또는 파일 경로로 교체 가능
    cap = cv2.VideoCapture(0)

    print("[VisionProcessor] 테스트 시작 (q: 종료)")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detected = processor.process_frame(frame)

        # 감지 결과 시각화
        viz_frame = frame.copy()
        for obj in detected:
            x, y, w, h = obj.bbox
            cv2.rectangle(viz_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"({obj.world_x:.1f}m, {obj.world_y:.1f}m)"
            cv2.putText(viz_frame, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("Vision Processor - Debug", viz_frame)
        topview = processor.get_topview_frame(frame)
        cv2.imshow("Top-view (IPM)", topview)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
