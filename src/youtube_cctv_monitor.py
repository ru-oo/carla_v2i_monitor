"""
youtube_cctv_monitor.py  [v2 — 전면 재작성]
============================================
원본 대비 주요 수정:
    ① MOG2 + 필터링 전면 강화  → 보행자·신호등·건물 오탐 제거
    ② IoU 기반 추적기          → CentroidTracker 교체 (ID 안정성 향상)
    ③ EMA 속도 평활화          → 순간 튀는 속도값 억제 / 상한 80km/h 클리핑
    ④ 2D 레이더 맵 수정        → IPM world 좌표 → 레이더 픽셀 올바르게 변환
    ⑤ 분류 캐시 가비지 컬렉션  → 사라진 트랙 캐시 누적 방지
    ⑥ 프레임 리사이즈 통일     → 해상도 불일치로 발생하는 좌표 오류 제거
    ⑦ ROI 다각형 튜닝 UI 추가  → 마우스 클릭으로 ROI 실시간 조정 가능

실행:
    python youtube_cctv_monitor.py --url "https://www.youtube.com/watch?v=XXXX"
    python youtube_cctv_monitor.py --file road.mp4
    python youtube_cctv_monitor.py --webcam 0

조작:
    q — 종료  |  s — 스크린샷  |  p — 일시정지  |  r — MOG2 배경 리셋
    [ / ] — ROI 상단 높이 조절  (↑낮게 / ↓높게)
"""

import argparse
import math
import os
import subprocess
import sys
import time
import traceback
from collections import deque
from typing import Tuple

import cv2
import numpy as np

# ── 로컬 모듈 임포트 ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from classifier import CLASSES, VehicleClassifier
    _CLS_OK = True
except ImportError as e:
    print(f"[경고] classifier 임포트 실패: {e}")
    _CLS_OK = False
    CLASSES = ["bus", "car", "truck", "van"]

from vision_processor import VisionProcessor, ROI_POLYGON, PIXELS_PER_METER


# =============================================
# 설정값
# =============================================
IMG_W = 1280
IMG_H = 720

DISP_W = 1900
DISP_H = 720

RADAR_W = 618   # 우측 패널 너비
RADAR_H = 362   # 2D 레이더 높이

# 차종 색상 (BGR)
COLORS = {
    "bus":     (0,   140, 255),
    "car":     (0,   210,  60),
    "truck":   (50,   50, 255),
    "van":     (255, 180,   0),
    "unknown": (150, 150, 150),
}
ICON = {"bus": "B", "car": "C", "truck": "T", "van": "V", "unknown": "?"}

# CNN 분류 주기 및 최소 BBox 크기
CLASSIFY_INTERVAL = 10    # N프레임마다 재분류
MIN_BBOX_CLASSIFY  = 30   # 이 픽셀 이하 크기는 분류 생략 ("car" 기본값)


# =============================================
# YouTube 스트림 URL 추출
# =============================================
def get_stream_url(youtube_url: str, res: str = "720") -> str:
    try:
        r = subprocess.run(
            ["yt-dlp", "--no-playlist",
             "-f", f"best[height<={res}][ext=mp4]/best[height<={res}]/best",
             "--get-url", youtube_url],
            capture_output=True, text=True, timeout=30,
        )
        url = r.stdout.strip().splitlines()[0]
        if not url:
            raise RuntimeError("스트림 URL 없음")
        print(f"[Stream] ≤{res}p 스트림 URL 획득")
        return url
    except FileNotFoundError:
        raise RuntimeError("yt-dlp 미설치: pip install yt-dlp")
    except subprocess.TimeoutExpired:
        raise RuntimeError("yt-dlp 타임아웃")


def open_capture(args) -> cv2.VideoCapture:
    if args.url:
        stream = get_stream_url(args.url, args.resolution)
        cap = cv2.VideoCapture(stream)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    elif args.file:
        if not os.path.exists(args.file):
            raise FileNotFoundError(f"파일 없음: {args.file}")
        cap = cv2.VideoCapture(args.file)
    else:
        cap = cv2.VideoCapture(args.webcam or 0)

    if not cap.isOpened():
        raise RuntimeError("VideoCapture 열기 실패")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src = args.url or args.file or f"webcam({args.webcam})"
    print(f"[Stream] {src}  |  {w}×{h}  FPS={fps:.1f}")
    return cap


# =============================================
# 2D 레이더 맵 (완전 재작성)
# =============================================
class RadarRenderer:
    """
    IPM world 좌표 (m) → 레이더 캔버스 픽셀 변환.

    world_x / world_y 는 IPM 400×400px 출력의 픽셀값을
    PIXELS_PER_METER 로 나눈 미터값.
    
    레이더 중심 = IPM 출력 중앙 (200/PPM, 200/PPM) 에 해당.
    """

    # IPM 출력 중심 (m) — VisionProcessor.PIXELS_PER_METER 를 단일 출처로 동기화
    # (과거 14.0으로 어긋나 있던 값을 import로 묶어 불일치를 없앰)
    _PPM = PIXELS_PER_METER        # = 18.18 (vision_processor와 동일)
    _IPM_CENTER_M = 200.0 / _PPM   # ≈ 11.0 m

    def __init__(self, w: int = RADAR_W, h: int = RADAR_H):
        self.w = w
        self.h = h
        # IPM world 범위: 0 ~ 400/PPM m → 레이더 전체 폭에 매핑
        self._world_range_x = 400.0 / self._PPM  # ≈ 22.0 m (IPM 400px = 22m)
        self._world_range_y = 400.0 / self._PPM
        self._scale_x = w / self._world_range_x   # px/m
        self._scale_y = h / self._world_range_y
        self._bg = self._build_bg()

    def _build_bg(self) -> np.ndarray:
        bg = np.full((self.h, self.w, 3), (10, 12, 20), dtype=np.uint8)
        cx, cy = self.w // 2, self.h // 2

        # 도로 배경
        hw_px_x = int(3.5 * self._scale_x * 2)  # 가정: 양방향 2차선 = 7m
        hw_px_y = int(3.5 * self._scale_y * 2)
        cv2.rectangle(bg, (0, cy - hw_px_y), (self.w, cy + hw_px_y), (30, 35, 48), -1)
        cv2.rectangle(bg, (cx - hw_px_x, 0), (cx + hw_px_x, self.h), (30, 35, 48), -1)

        # 거리 원호 (5m 단위)
        for r_m in [3, 6, 10, 14]:
            r_px = int(r_m * min(self._scale_x, self._scale_y))
            cv2.circle(bg, (cx, cy), r_px, (38, 52, 70), 1, cv2.LINE_AA)
            cv2.putText(bg, f"{r_m}m", (cx + r_px + 3, cy - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.26, (38, 52, 70), 1)

        # 도로 경계
        lane_col = (60, 80, 105)
        cv2.line(bg, (0, cy - hw_px_y), (self.w, cy - hw_px_y), lane_col, 1)
        cv2.line(bg, (0, cy + hw_px_y), (self.w, cy + hw_px_y), lane_col, 1)
        cv2.line(bg, (cx - hw_px_x, 0), (cx - hw_px_x, self.h), lane_col, 1)
        cv2.line(bg, (cx + hw_px_x, 0), (cx + hw_px_x, self.h), lane_col, 1)

        # 방향 레이블
        cv2.circle(bg, (cx, cy), 5, (0, 200, 255), -1)
        for txt, pos in [("N",(cx-5,14)),("S",(cx-5,self.h-4)),
                          ("E",(self.w-14,cy+5)),("W",(3,cy+5))]:
            cv2.putText(bg, txt, pos, cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, (0, 200, 255), 1)
        return bg

    def _world_to_radar(self, wx: float, wy: float) -> Tuple[int, int]:
        """IPM world(m) → 레이더 픽셀 (IPM 좌상=레이더 좌상)"""
        rx = int(wx * self._scale_x)
        ry = int(wy * self._scale_y)
        return rx, ry

    def render(self, tracked, counts: dict) -> np.ndarray:
        img = self._bg.copy()

        for trk in tracked:
            rx, ry = self._world_to_radar(trk.world_x, trk.world_y)
            if not (0 <= rx < self.w and 0 <= ry < self.h):
                continue

            vtype = trk.vehicle_type
            color = COLORS.get(vtype, COLORS["unknown"])

            # 차량 사각형 (실제 크기 비례)
            bw = max(4, int(1.8 * self._scale_x))
            bh = max(3, int(0.9 * self._scale_y))
            cv2.rectangle(img, (rx-bw, ry-bh), (rx+bw, ry+bh), color, -1)
            cv2.rectangle(img, (rx-bw, ry-bh), (rx+bw, ry+bh),
                          tuple(min(c+60,255) for c in color), 1)

            # 방향 화살표
            if trk.speed_kmh > 3.0:
                ang = math.radians(trk.direction_deg)
                ex = int(rx + 12 * math.cos(ang))
                ey = int(ry + 12 * math.sin(ang))
                if 0 <= ex < self.w and 0 <= ey < self.h:
                    cv2.arrowedLine(img, (rx, ry), (ex, ey),
                                    color, 1, cv2.LINE_AA, tipLength=0.4)

            # 트랙 ID
            cv2.putText(img, f"#{trk.track_id}",
                        (rx + bw + 2, ry + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.26, color, 1)

        # 타이틀
        cv2.putText(img, "2D RADAR MAP", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0,200,255), 2, cv2.LINE_AA)
        cv2.putText(img, "2D RADAR MAP", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0,220,255), 1, cv2.LINE_AA)

        # 우하단 범례
        self._draw_legend(img, counts)
        return img

    def _draw_legend(self, img, counts):
        pw, ph = 150, len(CLASSES) * 22 + 34
        px0 = self.w - pw - 5
        py0 = self.h - ph - 5
        roi = img[py0:py0+ph, px0:px0+pw]
        overlay = np.zeros_like(roi)
        cv2.addWeighted(roi, 0.3, overlay, 0.7, 0, roi)
        img[py0:py0+ph, px0:px0+pw] = roi
        cv2.rectangle(img, (px0,py0), (px0+pw,py0+ph), (30,40,60), 1)
        for i, cls in enumerate(CLASSES):
            col = COLORS[cls]
            iy  = py0 + 20 + i * 22
            cv2.rectangle(img, (px0+6,iy-8), (px0+20,iy+6), col, -1)
            cv2.putText(img, f"{cls.upper():<6}{counts.get(cls,0):>3}",
                        (px0+24, iy+4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1)
        total = sum(counts.values())
        cv2.putText(img, f"TOTAL  {total}",
                    (px0+6, py0+ph-7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200,200,200), 1)


# =============================================
# CCTV 뷰 오버레이
# =============================================
def draw_camera_view(frame: np.ndarray,
                     tracked,
                     cls_cache: dict,
                     fps: float,
                     src_label: str,
                     roi_polygon: np.ndarray) -> tuple:
    """검출 결과를 프레임에 오버레이하고 (annotated_frame, counts) 반환"""

    # ROI 경계 표시 (반투명)
    roi_overlay = frame.copy()
    cv2.polylines(roi_overlay, [roi_polygon], isClosed=True,
                  color=(0, 255, 120), thickness=2)
    cv2.addWeighted(roi_overlay, 0.7, frame, 0.3, 0, frame)

    counts = {c: 0 for c in CLASSES}

    for trk in tracked:
        x, y, w, h = trk.bbox
        vtype = cls_cache.get(trk.track_id, "unknown")
        color = COLORS.get(vtype, COLORS["unknown"])

        if vtype in counts:
            counts[vtype] += 1

        # BBox
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # 라벨 배경
        label = f"{ICON.get(vtype,'?')} {trk.speed_kmh:.0f}km/h"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        ly = max(y - 4, lh + 2)
        cv2.rectangle(frame, (x, ly-lh-2), (x+lw+4, ly+2), color, -1)
        cv2.putText(frame, label, (x+2, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,0,0), 1, cv2.LINE_AA)

        # 방향 화살표 (속도 > 3 km/h)
        if trk.speed_kmh > 3.0:
            ang = math.radians(trk.direction_deg)
            ex  = int(trk.pixel_x + 25 * math.cos(ang))
            ey  = int(trk.pixel_y + 25 * math.sin(ang))
            h_f, w_f = frame.shape[:2]
            if 0 <= ex < w_f and 0 <= ey < h_f:
                cv2.arrowedLine(frame, (trk.pixel_x, trk.pixel_y),
                                (ex, ey), color, 2, cv2.LINE_AA, tipLength=0.30)

    # ── 상단 HUD ─────────────────────────────────────────────
    hud = frame.copy()
    cv2.rectangle(hud, (0,0), (IMG_W, 44), (0,0,0), -1)
    cv2.addWeighted(hud, 0.55, frame, 0.45, 0, frame)

    lbl = os.path.basename(src_label) if os.path.exists(src_label) else src_label
    title = f"CCTV Road Monitor  |  {lbl[:60]}  |  FPS {fps:.1f}"
    cv2.putText(frame, title, (10, 29),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(frame, title, (10, 29),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (0, 200, 255), 1, cv2.LINE_AA)

    # ── 우상단 카운트 박스 ────────────────────────────────────
    bx = IMG_W - 140
    cv2.rectangle(frame, (bx-6, 48),
                  (IMG_W-4, 48+len(CLASSES)*24+26), (0,0,0), -1)
    for i, cls in enumerate(CLASSES):
        cv2.putText(frame, f"{cls.upper()}: {counts[cls]}",
                    (bx, 68+i*24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    COLORS[cls], 2, cv2.LINE_AA)
    cv2.putText(frame, f"TOTAL: {sum(counts.values())}",
                (bx, 68+len(CLASSES)*24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220,220,220), 1)

    return frame, counts


# =============================================
# 4분할 디스플레이 합성
# =============================================
def compose_display(cam: np.ndarray,
                    radar: np.ndarray,
                    topview: np.ndarray,
                    dbg: np.ndarray) -> np.ndarray:
    canvas = np.zeros((DISP_H, DISP_W, 3), dtype=np.uint8)

    # 좌측: CCTV (1280×720)
    h_c, w_c = cam.shape[:2]
    if w_c != IMG_W or h_c != IMG_H:
        cam = cv2.resize(cam, (IMG_W, IMG_H))
    canvas[:IMG_H, :IMG_W] = cam

    # 수직 구분선
    canvas[:, IMG_W:IMG_W+2] = (55, 60, 72)
    RX = IMG_W + 2
    RW = DISP_W - RX   # ≈ 618

    SEP = (55, 60, 72)

    # 우상단: 2D Radar (618×362)
    rad_s = cv2.resize(radar, (RW, RADAR_H))
    canvas[:RADAR_H, RX:RX+RW] = rad_s
    canvas[RADAR_H:RADAR_H+2, RX:] = SEP

    # 우중단: IPM Top-view (618×178)
    BEV_TOP = RADAR_H + 2
    BEV_H   = 178
    bev_sq  = BEV_H - 4
    tv = cv2.resize(topview, (bev_sq, bev_sq))
    bev_panel = np.zeros((BEV_H, RW, 3), dtype=np.uint8)
    bev_panel[2:2+bev_sq, 2:2+bev_sq] = tv
    cv2.putText(bev_panel, "[ IPM Top-view ]", (bev_sq+8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0,200,255), 1, cv2.LINE_AA)
    cv2.putText(bev_panel, "Inverse Perspective Mapping",
                (bev_sq+8, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, (100,120,140), 1, cv2.LINE_AA)
    canvas[BEV_TOP:BEV_TOP+BEV_H, RX:RX+RW] = bev_panel
    canvas[BEV_TOP+BEV_H:BEV_TOP+BEV_H+2, RX:] = SEP

    # 우하단: MOG2+Tracker 디버그
    DBG_TOP = BEV_TOP + BEV_H + 2
    DBG_H   = DISP_H - DBG_TOP
    if dbg is not None and DBG_H > 10:
        dbg_s = cv2.resize(dbg, (RW, DBG_H))
        canvas[DBG_TOP:DBG_TOP+DBG_H, RX:RX+RW] = dbg_s

    return canvas


# =============================================
# ROI 높이 실시간 조절
# =============================================
def adjust_roi_top(roi_poly: np.ndarray, delta: int) -> np.ndarray:
    """ROI 상단 Y좌표를 delta 픽셀만큼 이동 (두 상단 꼭짓점)"""
    new_poly = roi_poly.copy()
    new_poly[0, 1] = max(0, min(IMG_H-1, new_poly[0, 1] + delta))
    new_poly[1, 1] = max(0, min(IMG_H-1, new_poly[1, 1] + delta))
    return new_poly


# =============================================
# 메인 관제 루프
# =============================================
def run(args):
    cap = open_capture(args)
    src_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_label = args.url or args.file or f"Webcam({args.webcam})"

    # ROI 다각형 (조정 가능)
    roi_poly = ROI_POLYGON.copy()

    # 모듈 초기화
    vision = VisionProcessor(roi_polygon=roi_poly)
    radar  = RadarRenderer(w=RADAR_W, h=RADAR_H)

    classifier = None
    if _CLS_OK:
        try:
            classifier = VehicleClassifier()
        except FileNotFoundError as e:
            print(f"[경고] {e}")

    cls_cache: dict = {}   # {track_id: vehicle_type}
    frame_idx = 0
    fps       = 0.0
    t_last    = time.time()
    paused    = False

    # FPS 측정용 deque
    fps_buf = deque(maxlen=20)

    cv2.namedWindow("CCTV Road Monitor", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("CCTV Road Monitor", DISP_W, DISP_H)
    print("[Monitor] 시작  q=종료  s=스크린샷  p=일시정지  r=배경리셋  [/]=ROI조절")

    while True:
        # ── 일시정지 ────────────────────────────────────────
        if paused:
            key = cv2.waitKey(50) & 0xFF
            if key == ord("q"):   break
            elif key == ord("p"): paused = False
            continue

        ret, frame = cap.read()
        if not ret:
            if args.file:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                # MOG2 배경 리셋 (루프 재시작 시 배경 혼란 방지)
                vision.bg_sub = cv2.createBackgroundSubtractorMOG2(
                    history=500, varThreshold=40, detectShadows=False)
                continue
            break

        frame_idx += 1

        # ── 해상도 통일 (모든 좌표 계산의 기준) ─────────────
        if frame.shape[1] != IMG_W or frame.shape[0] != IMG_H:
            frame = cv2.resize(frame, (IMG_W, IMG_H))

        # ── Vision 처리 ──────────────────────────────────────
        tracked = vision.process_frame(frame)

        # ── CNN 차종 분류 ────────────────────────────────────
        if classifier is not None:
            for trk in tracked:
                needs_cls = (frame_idx % CLASSIFY_INTERVAL == 0
                             or trk.track_id not in cls_cache)
                if needs_cls:
                    x, y, w, h = trk.bbox
                    if w >= MIN_BBOX_CLASSIFY and h >= MIN_BBOX_CLASSIFY:
                        crop  = frame[max(0,y):min(IMG_H,y+h),
                                      max(0,x):min(IMG_W,x+w)]
                        if crop.size > 0:
                            vt = classifier.classify(crop)
                            cls_cache[trk.track_id] = vt
                        else:
                            cls_cache.setdefault(trk.track_id, "car")
                    else:
                        cls_cache.setdefault(trk.track_id, "car")
        else:
            for trk in tracked:
                cls_cache.setdefault(trk.track_id, "unknown")

        # 분류 캐시 가비지 컬렉션
        active_ids = {t.track_id for t in tracked}
        for old_id in list(cls_cache):
            if old_id not in active_ids:
                del cls_cache[old_id]

        # vehicle_type 동기화 (TrackedObject 에 반영)
        for trk in tracked:
            trk.vehicle_type = cls_cache.get(trk.track_id, "unknown")

        # ── 카운트 집계 ──────────────────────────────────────
        counts = {c: 0 for c in CLASSES}
        for trk in tracked:
            if trk.vehicle_type in counts:
                counts[trk.vehicle_type] += 1

        # ── 렌더링 ───────────────────────────────────────────
        cam_view, _ = draw_camera_view(
            frame.copy(), tracked, cls_cache, fps, src_label, roi_poly
        )
        radar_img   = radar.render(tracked, counts)
        topview_img = vision.get_topview_frame(frame)
        dbg_img     = vision.get_debug_image()

        output = compose_display(cam_view, radar_img, topview_img, dbg_img)

        # ── FPS ──────────────────────────────────────────────
        now = time.time()
        fps_buf.append(1.0 / max(now - t_last, 1e-6))
        fps    = sum(fps_buf) / len(fps_buf)
        t_last = now

        cv2.imshow("CCTV Road Monitor", output)

        # ── 키 입력 ──────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"):
            paused = True
            print("[Monitor] 일시정지")
        elif key == ord("s"):
            fname = f"screenshot_{int(time.time())}.jpg"
            cv2.imwrite(fname, output)
            print(f"[Monitor] 저장: {fname}")
        elif key == ord("r"):
            # MOG2 배경 모델 리셋
            vision.bg_sub = cv2.createBackgroundSubtractorMOG2(
                history=500, varThreshold=40, detectShadows=False)
            print("[Monitor] MOG2 배경 리셋")
        elif key == ord("["):
            roi_poly = adjust_roi_top(roi_poly, -10)
            vision.roi_polygon = roi_poly
            vision._roi_mask   = None
            print(f"[Monitor] ROI 상단 Y: {roi_poly[0,1]}")
        elif key == ord("]"):
            roi_poly = adjust_roi_top(roi_poly, +10)
            vision.roi_polygon = roi_poly
            vision._roi_mask   = None
            print(f"[Monitor] ROI 상단 Y: {roi_poly[0,1]}")

    cap.release()
    cv2.destroyAllWindows()
    print("[Monitor] 종료")


# =============================================
# CLI
# =============================================
def build_parser():
    p = argparse.ArgumentParser(
        description="YouTube / 로컬 영상 도로 CCTV 관제 시스템 v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python youtube_cctv_monitor.py --url "https://www.youtube.com/watch?v=XXXX"
  python youtube_cctv_monitor.py --file traffic.mp4
  python youtube_cctv_monitor.py --webcam 0
  python youtube_cctv_monitor.py --url "..." --resolution 480
        """,
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--url",    type=str, help="YouTube URL")
    src.add_argument("--file",   type=str, help="로컬 영상 파일")
    src.add_argument("--webcam", type=int, help="웹캠 ID")
    p.add_argument("--resolution", default="720",
                   help="YouTube 최대 해상도 (기본 720)")
    return p


def main():
    args = build_parser().parse_args()
    if not args.url and not args.file and args.webcam is None:
        args.webcam = 0
    try:
        run(args)
    except KeyboardInterrupt:
        print("\n[Monitor] 사용자 중단")
    except Exception:
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()