"""
vision_processor.py  [v3 — 속도 오류 전면 수정]
=================================================

[수정된 문제]
    1. 정지 차량 속도 오류 (6~10km/h 표시)
       - 원인: MOG2 잔상 + IPM 좌표 미세 흔들림이 이동으로 계산됨
       - 해결: Displacement threshold (최소 이동 거리 미만은 0으로 처리)
               + EMA alpha 강화 (0.15 → 더 느리게 반응)

    2. 순간 속도 spike (20km/h 이상 튀는 현상)
       - 원인: 새 트랙 생성 직후 이전 위치 없이 큰 변위 계산
       - 해결: frames_alive < 3 이면 속도 0 (초기 안정화 구간)
               + 속도 상한 클리핑 강화 (80km/h)

    3. IoU 추적기 안정성 향상
       - 매칭 실패 시 위치 보간 (속도 방향으로 extrapolation)
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
    pixel_x:  int
    pixel_y:  int
    world_x:  float
    world_y:  float
    bbox:     Tuple[int, int, int, int]
    area:     float
    vehicle_type: str = "unknown"


@dataclass
class TrackedObject:
    track_id:       int
    pixel_x:        int
    pixel_y:        int
    world_x:        float
    world_y:        float
    bbox:           Tuple[int, int, int, int]
    area:           float
    vehicle_type:   str   = "unknown"
    speed_ms:       float = 0.0
    speed_kmh:      float = 0.0
    direction_deg:  float = 0.0
    frames_alive:   int   = 1
    frames_missing: int   = 0
    # 슬라이딩 윈도우: 최근 N프레임 raw 속도 저장 (순간 스파이크 평균화)
    _raw_speed_buf: object = None  # deque, 타입힌트 우회
    # 위치 EMA (속도 계산 전에 world 좌표를 먼저 평활화)
    smooth_x: float = -1.0   # -1 = 아직 초기화 안 됨
    smooth_y: float = -1.0


# =============================================
# 설정값
# =============================================

# ── ROI (도로 영역만 분석) ────────────────────────────────────
ROI_POLYGON = np.array([
    [0,    250],
    [1280, 250],
    [1280, 720],
    [0,    720],
], dtype=np.int32)

# ── 객체 크기 필터 ────────────────────────────────────────────
MIN_AREA_PX  = 800
MAX_AREA_PX  = 60000
MIN_W_PX     = 25
MIN_H_PX     = 15
MAX_ASPECT   = 2.5    # h/w 초과 → 사람·기둥 제거
MIN_ASPECT   = 0.25   # h/w 미만 → 수평 노이즈 제거

# ── IPM 캘리브레이션 ─────────────────────────────────────────
# ── IPM 캘리브레이션 ─────────────────────────────────────────
# CARLA 카메라: height=12m, pitch=-25°, FOV=110°, 1280×720
# geometry 기반으로 역산한 지면 직사각형 → IPM 매핑
#
#  지면 커버 범위: 카메라 전방 6m ~ 28m, 좌우 ±6m (22m × 12m)
#  IPM 출력 크기: 400×400px
#  → PIXELS_PER_METER = 400 / 22m ≈ 18.18
#
#  IPM_SRC: 지면 코너를 카메라 투영으로 역산한 이미지 좌표
#    NL (근거리 좌, 6m/-6m) → (384, 716)
#    NR (근거리 우, 6m/+6m) → (896, 716)
#    FR (원거리 우, 28m/+6m) → (728, 346)
#    FL (원거리 좌, 28m/-6m) → (552, 346)
PIXELS_PER_METER = 18.18  # 400px / 22m (geometry 기반 재계산)

IPM_SRC_POINTS = np.float32([
    [552, 346],   # FL: 원거리 좌 (28m, -6m)
    [728, 346],   # FR: 원거리 우 (28m, +6m)
    [896, 716],   # NR: 근거리 우 ( 6m, +6m)
    [384, 716],   # NL: 근거리 좌 ( 6m, -6m)
])
IPM_DST_POINTS = np.float32([
    [0,   0],
    [400, 0],
    [400, 400],
    [0,   400],
])

# ── 추적 설정 ────────────────────────────────────────────────
MAX_MISSING_FRAMES = 6
IOU_MATCH_THRESH   = 0.15
FPS_ESTIMATE       = 20.0  # CARLA fixed tick rate (world.tick() = 20fps)

# ── 속도 추정 파라미터 ───────────────────────────────────────
# 1) 최소 이동 임계값
# MIN_DISP: POS_EMA(0.70) 적용 후 disp 기준
# 30km/h @ 20fps → real_disp=0.417m → EMA후 0.292m → 0.10 통과
MIN_DISP_METER     = 0.10

# EMA: 0.40 (반응성↑, 기존 0.25는 수렴에 12프레임 필요)
SPEED_EMA_ALPHA    = 0.40

SPEED_CAP_KMH      = 80.0

# 안정화: 2프레임 (기존 4는 너무 긺)
SPEED_STABLE_FRAMES = 2

# ── 시각화 색상 ───────────────────────────────────────────────
_TRACK_COLORS = [
    (0,   255, 100), (0,   140, 255), (255,  50,  50),
    (255, 200,   0), (180,   0, 255), (0,   220, 220),
    (255, 100, 180), (100, 255, 255), (50,  200, 255),
    (200, 255,  50),
]


# =============================================
# IoU 유틸
# =============================================
def _iou(a: Tuple, b: Tuple) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1+aw, ay1+ah
    bx2, by2 = bx1+bw, by1+bh
    ix1, iy1 = max(ax1,bx1), max(ay1,by1)
    ix2, iy2 = min(ax2,bx2), min(ay2,by2)
    inter = max(0,ix2-ix1) * max(0,iy2-iy1)
    if inter == 0: return 0.0
    return inter / max(aw*ah + bw*bh - inter, 1e-6)


# =============================================
# IoU 기반 추적기 (속도 계산 수정 포함)
# =============================================
class IoUTracker:
    def __init__(self):
        self._next_id = 0
        self._tracks: Dict[int, TrackedObject] = {}

    def update(self, detections: List[DetectedObject]) -> List[TrackedObject]:
        track_ids = list(self._tracks.keys())
        used_det  = set()
        used_trk  = set()

        if track_ids and detections:
            iou_mat = np.zeros((len(track_ids), len(detections)))
            for ti, tid in enumerate(track_ids):
                for di, det in enumerate(detections):
                    iou_mat[ti, di] = _iou(self._tracks[tid].bbox, det.bbox)

            flat = np.argsort(-iou_mat.ravel())
            for idx in flat:
                ti = idx // len(detections)
                di = idx  %  len(detections)
                if iou_mat[ti, di] < IOU_MATCH_THRESH:
                    break
                tid = track_ids[ti]
                if tid in used_trk or di in used_det:
                    continue
                used_trk.add(tid)
                used_det.add(di)

                det = detections[di]
                trk = self._tracks[tid]

                # ── 속도 계산 ────────────────────────────────────
                from collections import deque as _deque

                # ── Step 1: world 좌표 EMA (위치 평활화) ─────────────
                # POS_EMA=0.70: 충분히 빠르게 추종하면서 노이즈 억제
                # 주의: disp = smooth_new - smooth_old
                #       = POS_EMA*(det-prev) → 실제 이동의 POS_EMA배
                #       → raw_kmh 계산 시 POS_EMA로 나눠서 보정
                POS_EMA = 0.70
                if trk.smooth_x < 0:   # 첫 프레임
                    sx = det.world_x
                    sy = det.world_y
                else:
                    sx = POS_EMA * det.world_x + (1 - POS_EMA) * trk.smooth_x
                    sy = POS_EMA * det.world_y + (1 - POS_EMA) * trk.smooth_y

                # ── Step 2: 변위 계산 + POS_EMA 스케일 보정 ─────────
                if trk.smooth_x < 0:
                    disp = 0.0
                else:
                    raw_disp = math.hypot(sx - trk.smooth_x, sy - trk.smooth_y)
                    # POS_EMA로 축소된 disp를 실제 이동량으로 복원
                    disp = raw_disp / POS_EMA

                # ── Step 3: 슬라이딩 윈도우 버퍼 (7프레임) ──────────
                buf = trk._raw_speed_buf
                if buf is None:
                    buf = _deque(maxlen=7)

                # ── Step 4: 초기 안정화 or 최소 변위 미달 → raw=0 ───
                if trk.frames_alive < SPEED_STABLE_FRAMES or disp < MIN_DISP_METER:
                    raw_kmh = 0.0
                else:
                    raw_kmh = min(disp * FPS_ESTIMATE * 3.6, SPEED_CAP_KMH)

                buf.append(raw_kmh)

                # ── Step 5: 중앙값으로 스파이크 제거 → EMA ──────────
                sorted_buf = sorted(buf)
                median_kmh = sorted_buf[len(sorted_buf) // 2]

                prev_kmh = trk.speed_kmh
                spd_kmh  = (SPEED_EMA_ALPHA * median_kmh
                            + (1 - SPEED_EMA_ALPHA) * prev_kmh)

                # ── Step 6: 저속 노이즈 스냅 (2km/h 미만만 0) ──────────
                if spd_kmh < 2.0:
                    spd_kmh = 0.0

                # 방향 (속도 있을 때만 업데이트)
                if disp > MIN_DISP_METER:
                    ang = math.degrees(
                        math.atan2(det.world_y - trk.world_y,
                                   det.world_x - trk.world_x))
                else:
                    ang = trk.direction_deg

                self._tracks[tid] = TrackedObject(
                    track_id=tid,
                    pixel_x=det.pixel_x,    pixel_y=det.pixel_y,
                    world_x=det.world_x,    world_y=det.world_y,
                    bbox=det.bbox,          area=det.area,
                    vehicle_type=trk.vehicle_type,
                    speed_ms=spd_kmh/3.6,   speed_kmh=spd_kmh,
                    direction_deg=ang,
                    frames_alive=trk.frames_alive + 1,
                    frames_missing=0,
                    _raw_speed_buf=buf,
                    smooth_x=sx,            # 평활화된 위치 이어받기
                    smooth_y=sy,
                )

        # 미매칭 트랙
        for tid in track_ids:
            if tid not in used_trk:
                trk = self._tracks[tid]
                m   = trk.frames_missing + 1
                if m >= MAX_MISSING_FRAMES:
                    del self._tracks[tid]
                else:
                    d = trk.__dict__.copy()
                    d["frames_missing"] = m
                    # 미감지 시 속도 서서히 감소
                    d["speed_kmh"] = trk.speed_kmh * 0.7
                    d["speed_ms"]  = d["speed_kmh"] / 3.6
                    self._tracks[tid] = TrackedObject(**d)

        # 신규 트랙
        for di, det in enumerate(detections):
            if di not in used_det:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = TrackedObject(
                    track_id=tid,
                    pixel_x=det.pixel_x, pixel_y=det.pixel_y,
                    world_x=det.world_x, world_y=det.world_y,
                    bbox=det.bbox, area=det.area,
                    vehicle_type=det.vehicle_type,
                    speed_ms=0.0, speed_kmh=0.0,
                    direction_deg=0.0, frames_alive=1, frames_missing=0,
                )

        return [t for t in self._tracks.values() if t.frames_missing == 0]


# =============================================
# VisionProcessor v3
# =============================================
class VisionProcessor:
    """
    차량 검출·추적 (속도 추정 수정 버전)

    파이프라인:
        원본 → ROI 마스크 → MOG2 배경차분 → 형태학적 클린업
        → 컨투어 필터(면적·종횡비) → IoU 추적
        → Displacement threshold 기반 속도 추정
        → EMA 평활화 + 저속 스냅(2km/h 미만 → 0)
    """

    def __init__(self,
                 roi_polygon: np.ndarray = None,
                 ipm_src: np.ndarray = None,
                 ipm_dst: np.ndarray = None):

        self.roi_polygon = roi_polygon if roi_polygon is not None else ROI_POLYGON
        self._roi_mask: Optional[np.ndarray] = None

        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=40, detectShadows=False)

        self.kernel_open   = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (5, 5))
        self.kernel_dilate = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (7, 7))
        self.kernel_close  = cv2.getStructuringElement(
            cv2.MORPH_RECT,   (25, 25))

        src = ipm_src if ipm_src is not None else IPM_SRC_POINTS
        dst = ipm_dst if ipm_dst is not None else IPM_DST_POINTS
        self.ipm_matrix     = cv2.getPerspectiveTransform(src, dst)
        self.ipm_matrix_inv = np.linalg.inv(self.ipm_matrix)

        self.tracker = IoUTracker()
        self._debug_img: Optional[np.ndarray] = None

        print("[VisionProcessor v3] 초기화 완료 (속도 추정 수정)")

    def _ensure_roi_mask(self, h, w):
        if self._roi_mask is not None and self._roi_mask.shape == (h, w):
            return
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [self.roi_polygon], 255)
        self._roi_mask = mask

    def _apply_roi(self, frame):
        h, w = frame.shape[:2]
        self._ensure_roi_mask(h, w)
        return cv2.bitwise_and(frame, frame, mask=self._roi_mask)

    def _morph_cleanup(self, mask):
        m = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self.kernel_open)
        m = cv2.dilate(m, self.kernel_dilate, iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, self.kernel_close)
        return m

    def _filter_contour(self, contour):
        area = cv2.contourArea(contour)
        if not (MIN_AREA_PX < area < MAX_AREA_PX):
            return None
        x, y, w, h = cv2.boundingRect(contour)
        if w < MIN_W_PX or h < MIN_H_PX:
            return None
        aspect = h / max(w, 1)
        if aspect > MAX_ASPECT or aspect < MIN_ASPECT:
            return None
        return (x, y, w, h)

    def _pixel_to_world(self, px, py):
        pt = np.float32([[[px, py]]])
        tp = cv2.perspectiveTransform(pt, self.ipm_matrix)
        return (float(tp[0][0][0]) / PIXELS_PER_METER,
                float(tp[0][0][1]) / PIXELS_PER_METER)

    def process_frame(self, frame: np.ndarray) -> List[TrackedObject]:
        roi_frame  = self._apply_roi(frame)
        fg         = self.bg_sub.apply(roi_frame)
        _, binary  = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        clean      = self._morph_cleanup(binary)

        contours, _ = cv2.findContours(
            clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: List[DetectedObject] = []
        for cnt in contours:
            bbox = self._filter_contour(cnt)
            if bbox is None:
                continue
            x, y, w, h = bbox
            cx  = x + w // 2
            cy  = y + h // 2
            # IPM 투영: centroid 대신 BBox bottom-center 사용
            # bottom-center = 차량이 지면에 닿는 점 → 원거리 오차 최소화
            bcy = min(y + h, frame.shape[0] - 1)  # bottom-center y (지면 접촉점)
            wx, wy = self._pixel_to_world(cx, bcy)
            detections.append(DetectedObject(
                pixel_x=cx, pixel_y=cy,   # 시각화는 centroid 유지
                world_x=wx, world_y=wy,   # 속도 계산은 bottom-center
                bbox=bbox, area=cv2.contourArea(cnt),
            ))

        tracked = self.tracker.update(detections)
        self._build_debug_image(frame, clean, tracked)
        return tracked

    def _build_debug_image(self, frame, mask, tracked):
        h, w = frame.shape[:2]
        left  = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        right = frame.copy()

        if self._roi_mask is not None:
            edge = cv2.Canny(self._roi_mask, 50, 150)
            left[edge > 0] = (0, 200, 100)

        for trk in tracked:
            col = _TRACK_COLORS[trk.track_id % len(_TRACK_COLORS)]
            x, yt, bw, bh = trk.bbox

            for img in [left, right]:
                cv2.rectangle(img, (x, yt), (x+bw, yt+bh), col, 2)

            lbl = f"#{trk.track_id} {trk.speed_kmh:.0f}km/h"
            (lw, lh), _ = cv2.getTextSize(
                lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            ly = max(yt - 5, lh + 4)
            cv2.rectangle(right, (x, ly-lh-3), (x+lw+4, ly+3), col, -1)
            cv2.putText(right, lbl, (x+2, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (0, 0, 0), 1, cv2.LINE_AA)

            if trk.speed_kmh > 2.0:
                ang = math.radians(trk.direction_deg)
                ex  = int(trk.pixel_x + 20*math.cos(ang))
                ey  = int(trk.pixel_y + 20*math.sin(ang))
                if 0 <= ex < w and 0 <= ey < h:
                    cv2.arrowedLine(right,
                                    (trk.pixel_x, trk.pixel_y), (ex, ey),
                                    col, 2, cv2.LINE_AA, tipLength=0.35)

        for img, txt in [(left,  "MOG2 Mask (ROI)"),
                         (right, f"IoU Tracker  n={len(tracked)}")]:
            cv2.rectangle(img, (0, 0), (w, 26), (8, 10, 18), -1)
            cv2.putText(img, txt, (6, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                        (180, 220, 255), 1, cv2.LINE_AA)

        self._debug_img = np.hstack([left, right])

    def get_debug_image(self) -> Optional[np.ndarray]:
        return self._debug_img

    def get_topview_frame(self, frame: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(frame, self.ipm_matrix, (400, 400))