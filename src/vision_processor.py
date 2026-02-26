"""
vision_processor.py
===================
담당: 김진수 (컴퓨터 비전 & OpenCV 알고리즘)
브랜치: feature/vision-logic

역할:
    - MOG2 배경 차분으로 움직이는 차량 마스크 추출
    - 형태학적 연산으로 노이즈 제거
    - IPM(Inverse Perspective Mapping)으로 2D → Top-view 3D 좌표 변환
    - CentroidTracker로 프레임 간 객체 추적 + 속도 추정
    - 감지된 객체의 ROI(Region of Interest) 좌표 반환

산출물:
    - 감지된 차량의 TrackedObject 리스트 (track_id, speed_kmh, direction 포함)
    - get_debug_image(): PPT 발표용 디버그 패널 시각화 이미지
"""

import math
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# =============================================
# 데이터 구조
# =============================================
@dataclass
class DetectedObject:
    """감지된 객체 정보 (단일 프레임, 추적 이전)"""
    pixel_x: int            # 원본 이미지 픽셀 X
    pixel_y: int            # 원본 이미지 픽셀 Y
    world_x: float          # Top-view 변환 후 물리적 X (m)
    world_y: float          # Top-view 변환 후 물리적 Y (m)
    bbox: Tuple[int, int, int, int]   # (x, y, w, h) Bounding Box
    area: float             # 마스크 내 픽셀 면적
    vehicle_type: str = "unknown"    # CNN 분류 결과


@dataclass
class TrackedObject:
    """추적 중인 객체 정보 (프레임 간 연속성 포함)"""
    track_id:      int
    pixel_x:       int
    pixel_y:       int
    world_x:       float
    world_y:       float
    bbox:          Tuple[int, int, int, int]
    area:          float
    vehicle_type:  str   = "unknown"
    speed_ms:      float = 0.0      # 추정 속도 (m/s)
    speed_kmh:     float = 0.0      # 추정 속도 (km/h)
    direction_deg: float = 0.0      # 이동 방향 (도, 0=동쪽, CCW)
    frames_alive:  int   = 1        # 추적 유지 프레임 수
    frames_missing:int   = 0        # 연속 미감지 프레임 수


# =============================================
# 설정값 (카메라 캘리브레이션에 맞게 수정 필요)
# =============================================

# IPM 변환용 소스 포인트 (카메라 시야 내 도로 4개 꼭짓점, 픽셀 단위)
IPM_SRC_POINTS = np.float32([
    [320, 400],   # 좌상단
    [960, 400],   # 우상단
    [1200, 700],  # 우하단
    [80, 700],    # 좌하단
])

# IPM 변환용 목적지 포인트 (20m x 20m 교차로 → 400x400 px)
IPM_DST_POINTS = np.float32([
    [0, 0],
    [400, 0],
    [400, 400],
    [0, 400],
])

PIXELS_PER_METER  = 20.0   # IPM 출력에서 1m당 픽셀 수
MIN_CONTOUR_AREA  = 300    # 최소 객체 면적 (노이즈 필터)
MAX_CONTOUR_AREA  = 60000  # 최대 객체 면적 (이상치 필터)

# ── 추적 설정 ────────────────────────────────────
MAX_MISSING_FRAMES = 8     # 이 프레임 이상 미감지 시 트랙 삭제
MAX_MATCH_DIST_PX  = 80    # 중심점 매칭 최대 픽셀 거리
FPS_ESTIMATE       = 20.0  # 속도 추정용 FPS (CARLA sync 20Hz)

# ── 시각화 색상 (BGR, 트랙 ID별 순환) ─────────────
_TRACK_COLORS = [
    (0,   255, 100),   # 초록
    (0,   140, 255),   # 주황
    (255,  50,  50),   # 파랑
    (255, 200,   0),   # 하늘
    (180,   0, 255),   # 보라
    (0,   220, 220),   # 노랑
    (255, 100, 180),   # 분홍
    (100, 255, 255),   # 연초록
]


# =============================================
# 중심점 기반 다중 객체 추적기
# =============================================
class CentroidTracker:
    """
    그리디 중심점(centroid) 매칭 기반 다중 객체 추적기

    알고리즘:
        1. 현재 프레임 감지 → 중심점 리스트
        2. 기존 트랙과 거리 행렬 계산
        3. 그리디 매칭 (최단 거리 페어링)
        4. 미매칭 감지 → 새 트랙 생성
        5. 장기 미감지 트랙 → 삭제
        6. 속도·방향 = 월드 좌표 변위 × FPS
    """

    def __init__(self):
        self._next_id   = 0
        self._tracks:   Dict[int, TrackedObject] = {}
        self._prev_cxy: Dict[int, Tuple[int, int]] = {}   # track_id → (px, py)

    def update(self, detections: List[DetectedObject]) -> List[TrackedObject]:
        """
        새 감지 결과로 트랙 업데이트.

        Returns:
            현재 활성 TrackedObject 리스트 (frames_missing == 0)
        """
        track_ids = list(self._tracks.keys())
        used_det  = set()
        used_trk  = set()

        # ── 거리 행렬 & 그리디 매칭 ──────────────────────────
        if track_ids and detections:
            dist_mat = np.full((len(track_ids), len(detections)), np.inf)
            for ti, tid in enumerate(track_ids):
                pcx, pcy = self._prev_cxy.get(
                    tid,
                    (self._tracks[tid].pixel_x, self._tracks[tid].pixel_y)
                )
                for di, det in enumerate(detections):
                    dist_mat[ti, di] = math.hypot(
                        det.pixel_x - pcx, det.pixel_y - pcy
                    )

            for _ in range(min(len(track_ids), len(detections))):
                if dist_mat.size == 0:
                    break
                idx = np.unravel_index(np.argmin(dist_mat), dist_mat.shape)
                ti, di = int(idx[0]), int(idx[1])
                if dist_mat[ti, di] > MAX_MATCH_DIST_PX:
                    break
                used_trk.add(track_ids[ti])
                used_det.add(di)
                dist_mat[ti, :] = np.inf
                dist_mat[:, di] = np.inf

                # 트랙 업데이트 + 속도 추정
                tid = track_ids[ti]
                det = detections[di]
                trk = self._tracks[tid]

                dw  = math.hypot(det.world_x - trk.world_x,
                                 det.world_y - trk.world_y)
                spd = min(dw * FPS_ESTIMATE, 30.0)   # 108 km/h cap
                ang = math.degrees(
                    math.atan2(det.world_y - trk.world_y,
                               det.world_x - trk.world_x)
                )

                self._tracks[tid] = TrackedObject(
                    track_id=tid,
                    pixel_x=det.pixel_x,    pixel_y=det.pixel_y,
                    world_x=det.world_x,    world_y=det.world_y,
                    bbox=det.bbox,          area=det.area,
                    vehicle_type=trk.vehicle_type,
                    speed_ms=spd,           speed_kmh=spd * 3.6,
                    direction_deg=ang,
                    frames_alive=trk.frames_alive + 1,
                    frames_missing=0,
                )
                self._prev_cxy[tid] = (det.pixel_x, det.pixel_y)

        # ── 미매칭 기존 트랙: missing 증가 또는 삭제 ─────────
        for tid in track_ids:
            if tid not in used_trk:
                trk     = self._tracks[tid]
                missing = trk.frames_missing + 1
                if missing >= MAX_MISSING_FRAMES:
                    del self._tracks[tid]
                    self._prev_cxy.pop(tid, None)
                else:
                    old = trk.__dict__.copy()
                    old["frames_missing"] = missing
                    self._tracks[tid] = TrackedObject(**old)

        # ── 미매칭 새 감지: 새 트랙 생성 ──────────────────────
        for di, det in enumerate(detections):
            if di not in used_det:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = TrackedObject(
                    track_id=tid,
                    pixel_x=det.pixel_x,    pixel_y=det.pixel_y,
                    world_x=det.world_x,    world_y=det.world_y,
                    bbox=det.bbox,          area=det.area,
                    vehicle_type=det.vehicle_type,
                    speed_ms=0.0,           speed_kmh=0.0,
                    direction_deg=0.0,
                    frames_alive=1,         frames_missing=0,
                )
                self._prev_cxy[tid] = (det.pixel_x, det.pixel_y)

        return [t for t in self._tracks.values() if t.frames_missing == 0]


# =============================================
# VisionProcessor (업그레이드 버전)
# =============================================
class VisionProcessor:
    """
    V2I 비전 처리 메인 클래스 (업그레이드)

    MOG2 배경 차분 + IPM 좌표 변환 + CentroidTracker 추적

    추가 기능:
        - CentroidTracker: 프레임 간 ID 유지, 속도·방향 추정
        - get_debug_image(): PPT 발표용 마스크 + 추적 오버레이 시각화
    """

    def __init__(self):
        # MOG2 배경 차분기 (더 빠른 학습, 높은 민감도)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200,
            varThreshold=40,
            detectShadows=True,
        )

        # IPM 변환 행렬
        self.ipm_matrix     = cv2.getPerspectiveTransform(
            IPM_SRC_POINTS, IPM_DST_POINTS
        )
        self.ipm_matrix_inv = np.linalg.inv(self.ipm_matrix)

        # 형태학적 연산 커널
        self.kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

        # 추적기
        self.tracker = CentroidTracker()

        # 디버그 이미지 캐시
        self._debug_img: Optional[np.ndarray] = None

        print("[VisionProcessor] 초기화 완료 (MOG2 + CentroidTracker)")

    # ── 처리 단계 ────────────────────────────────────────────────

    def subtract_background(self, frame: np.ndarray) -> np.ndarray:
        fg_mask = self.bg_subtractor.apply(frame)
        _, binary_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        return binary_mask

    def remove_noise(self, mask: np.ndarray) -> np.ndarray:
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self.kernel_open)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self.kernel_close)
        return closed

    def find_vehicle_contours(self, clean_mask: np.ndarray) -> List:
        contours, _ = cv2.findContours(
            clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return [
            c for c in contours
            if MIN_CONTOUR_AREA < cv2.contourArea(c) < MAX_CONTOUR_AREA
        ]

    def pixel_to_world(self, px: int, py: int) -> Tuple[float, float]:
        pt  = np.float32([[[px, py]]])
        tp  = cv2.perspectiveTransform(pt, self.ipm_matrix)
        wx  = float(tp[0][0][0]) / PIXELS_PER_METER
        wy  = float(tp[0][0][1]) / PIXELS_PER_METER
        return wx, wy

    # ── 메인 처리 (process_frame) ────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> List[TrackedObject]:
        """
        단일 프레임 처리 → TrackedObject 리스트 반환.
        내부적으로 디버그 이미지도 생성 (get_debug_image()로 접근).

        Args:
            frame: BGR numpy 배열 (카메라 프레임)

        Returns:
            tracked: 현재 활성 TrackedObject 리스트
        """
        mask       = self.subtract_background(frame)
        clean_mask = self.remove_noise(mask)
        contours   = self.find_vehicle_contours(clean_mask)

        detections: List[DetectedObject] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cx, cy      = x + w // 2, y + h // 2
            wx, wy      = self.pixel_to_world(cx, cy)
            detections.append(DetectedObject(
                pixel_x=cx, pixel_y=cy,
                world_x=wx, world_y=wy,
                bbox=(x, y, w, h),
                area=cv2.contourArea(contour),
            ))

        tracked = self.tracker.update(detections)
        self._build_debug_image(frame, clean_mask, tracked)
        return tracked

    # ── 디버그 패널 생성 ─────────────────────────────────────────

    def _build_debug_image(self,
                           frame:   np.ndarray,
                           mask:    np.ndarray,
                           tracked: List[TrackedObject]):
        """
        PPT 발표용 디버그 이미지 생성:
            좌측: MOG2 이진 마스크 (컬러라이즈)
            우측: 원본 프레임 + 추적 오버레이 (BBox, ID, 속도, 방향)
        """
        h, w = frame.shape[:2]

        # ── 마스크 컬러라이즈 ─────────────────────────────────
        mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_PLASMA)

        # ── 추적 오버레이 (원본 위) ───────────────────────────
        overlay = frame.copy()
        for trk in tracked:
            tid   = trk.track_id
            color = _TRACK_COLORS[tid % len(_TRACK_COLORS)]
            x, y_t, bw, bh = trk.bbox

            # Bounding Box
            cv2.rectangle(overlay, (x, y_t), (x + bw, y_t + bh), color, 2)

            # 레이블 (ID, 속도)
            label = f"#{tid}  {trk.speed_kmh:.0f}km/h"
            (lw, lh), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1
            )
            ly = max(y_t - 6, lh + 4)
            cv2.rectangle(overlay,
                          (x, ly - lh - 3), (x + lw + 4, ly + 3),
                          color, -1)
            cv2.putText(overlay, label, (x + 2, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (0, 0, 0), 1, cv2.LINE_AA)

            # 방향 화살표
            cx, cy_ = trk.pixel_x, trk.pixel_y
            ang = math.radians(trk.direction_deg)
            ex  = int(cx + 22 * math.cos(ang))
            ey  = int(cy_ + 22 * math.sin(ang))
            if 0 <= ex < w and 0 <= ey < h and trk.speed_kmh > 1.0:
                cv2.arrowedLine(overlay, (cx, cy_), (ex, ey),
                                color, 2, cv2.LINE_AA, tipLength=0.35)

        # ── 마스크에도 BBox 오버레이 (흰 테두리) ─────────────
        for trk in tracked:
            tid   = trk.track_id
            color = _TRACK_COLORS[tid % len(_TRACK_COLORS)]
            x, y_t, bw, bh = trk.bbox
            cv2.rectangle(mask_colored, (x, y_t), (x + bw, y_t + bh), color, 2)
            cv2.putText(mask_colored, f"#{tid}",
                        (x, y_t - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.40, color, 1, cv2.LINE_AA)

        # ── 상단 정보 HUD ─────────────────────────────────────
        for img in [mask_colored, overlay]:
            cv2.rectangle(img, (0, 0), (w, 28), (10, 12, 20), -1)

        cv2.putText(mask_colored, "MOG2 Foreground Mask",
                    (8, 19), cv2.FONT_HERSHEY_SIMPLEX,
                    0.50, (200, 200, 80), 1, cv2.LINE_AA)
        cv2.putText(overlay,
                    f"CentroidTracker  tracks={len(tracked)}",
                    (8, 19), cv2.FONT_HERSHEY_SIMPLEX,
                    0.50, (0, 200, 200), 1, cv2.LINE_AA)

        # ── 좌우 합성 ─────────────────────────────────────────
        self._debug_img = np.hstack([mask_colored, overlay])

    def get_debug_image(self) -> Optional[np.ndarray]:
        """현재 프레임 디버그 이미지 반환 (마스크 | 추적 오버레이)"""
        return self._debug_img

    def get_topview_frame(self, frame: np.ndarray) -> np.ndarray:
        """원본 프레임 → Top-view(IPM) 변환 (시각화용)"""
        return cv2.warpPerspective(frame, self.ipm_matrix, (400, 400))


# =============================================
# 단독 실행 테스트 (개발·디버깅용)
# =============================================
def main():
    """웹캠 또는 영상 파일로 알고리즘 단독 테스트"""
    processor = VisionProcessor()
    cap       = cv2.VideoCapture(0)

    print("[VisionProcessor] 테스트 시작 (q: 종료)")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        tracked = processor.process_frame(frame)

        # 원본 프레임 오버레이
        viz = frame.copy()
        for trk in tracked:
            x, y, w, h = trk.bbox
            color = _TRACK_COLORS[trk.track_id % len(_TRACK_COLORS)]
            cv2.rectangle(viz, (x, y), (x + w, y + h), color, 2)
            label = (f"#{trk.track_id} "
                     f"({trk.world_x:.1f}m,{trk.world_y:.1f}m) "
                     f"{trk.speed_kmh:.0f}km/h")
            cv2.putText(viz, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        cv2.imshow("Vision Processor - Tracked", viz)

        dbg = processor.get_debug_image()
        if dbg is not None:
            dbg_small = cv2.resize(dbg, (dbg.shape[1] // 2, dbg.shape[0] // 2))
            cv2.imshow("MOG2 + CentroidTracker", dbg_small)

        topview = processor.get_topview_frame(frame)
        cv2.imshow("Top-view (IPM)", topview)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
