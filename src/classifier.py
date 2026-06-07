"""
classifier.py  [v2 — 수업 모델 비교 + 발표용 결과 출력]
==========================================================

[수업에서 배운 내용 반영]
    - Gradient Descent (AdamW 옵티마이저 — loss.backward() + step())
    - nn.Sequential 기반 Linear 모델 (MLP)
    - Multi-class Classification (Softmax + CrossEntropyLoss)
    - CNN 모델과 동일 조건(데이터·에폭·LR)으로 학습 후 비교

[문제 수정]
    1. 라벨 플리커링 해결
       - classify_with_conf() → (label, confidence) 반환
       - confidence < CONF_THRESHOLD 이면 "uncertain" 반환
       - 호출자(monitor)에서 투표 캐시로 라벨 안정화

    2. 발표용 결과 자동 저장 (results/ 폴더)
       - 01_learning_curves.png  (CNN vs Linear, Loss / Val Acc 4패널)
       - 02_confusion_matrix.png (CNN / Linear 나란히)
       - 03_model_comparison.png (Accuracy, F1, 파라미터, 추론속도 비교)

[실행]
    python src/classifier.py          # CNN + Linear 동시 학습 + 결과 저장
    python src/classifier.py --cnn    # CNN만 학습
    python src/classifier.py --linear # Linear만 학습
"""

import argparse
import os
import random
import time
from collections import Counter

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score, accuracy_score)

# =============================================
# 경로 설정
# =============================================
_SRC_DIR    = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.dirname(_SRC_DIR)
DATA_DIR    = os.path.join(BASE_DIR, "data", "vehicle_images")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
RESULT_DIR  = os.path.join(BASE_DIR, "results")

CNN_PATH    = os.path.join(MODEL_DIR, "classifier_cnn.pth")
LINEAR_PATH = os.path.join(MODEL_DIR, "classifier_linear.pth")
MODEL_PATH  = CNN_PATH  # 하위 호환

for _d in (MODEL_DIR, RESULT_DIR):
    os.makedirs(_d, exist_ok=True)

# =============================================
# 하이퍼파라미터
# =============================================
IMG_SIZE       = 64
BATCH_SIZE     = 64
EPOCHS         = 30
LEARNING_RATE  = 1e-3
WEIGHT_DECAY   = 1e-4
PATIENCE       = 7

CLASSES        = ["bus", "car", "truck", "van"]
NUM_CLASSES    = len(CLASSES)
CONF_THRESHOLD = 0.55  # 신뢰도 임계값

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


# =============================================
# ① CNN 모델 (4-Block + GAP)
# =============================================
class VehicleCNN(nn.Module):
    """
    경량 4-Block CNN
    Block 1~4 : Conv3x3(x2) + BN + ReLU + MaxPool + Dropout2D
    GAP       : (B,256,4,4) -> (B,256)
    Head      : Linear 256->128->NUM_CLASSES
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        def _block(ic, oc):
            return nn.Sequential(
                nn.Conv2d(ic, oc, 3, padding=1, bias=False),
                nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
                nn.Conv2d(oc, oc, 3, padding=1, bias=False),
                nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(0.1),
            )

        self.features = nn.Sequential(
            _block(3,   32),
            _block(32,  64),
            _block(64,  128),
            _block(128, 256),
        )
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(self.gap(self.features(x)))

    @property
    def model_name(self): return "CNN (4-Block)"

    @property
    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================
# ② Linear 모델 — 수업 방식 (nn.Sequential)
# =============================================
class VehicleLinear(nn.Module):
    """
    수업에서 배운 방식으로 구현한 Linear 분류기

    ── 수업 개념 반영 ──────────────────────────────────────
    model = nn.Sequential(...)          # 수업에서 배운 구성 방식
    loss  = CrossEntropyLoss(...)       # Multi-class Classification
    optimizer.zero_grad()               # Gradient 초기화
    loss.backward()                     # Gradient Descent 역전파
    optimizer.step()                    # 파라미터 업데이트
    ───────────────────────────────────────────────────────

    CNN과의 차이:
        CNN    - 합성곱으로 공간적 특징(엣지, 텍스처, 형태) 계층 추출
        Linear - 픽셀을 1D로 펼쳐 전결합층만 사용 (공간 구조 무시)
        → 이 차이로 CNN이 차종 분류에서 우수함을 발표에서 정량 비교
    """

    def __init__(self, num_classes: int = NUM_CLASSES,
                 img_size: int = IMG_SIZE):
        super().__init__()
        flat = 3 * img_size * img_size  # 12288

        # ── 수업 방식: nn.Sequential로 모델 구성 ──────────────
        self.model = nn.Sequential(
            nn.Flatten(),               # 3x64x64 -> 12288
            nn.Linear(flat, 1024),      # 수업: y = W*x + b
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
            # CrossEntropyLoss 내부에 Softmax 포함
            # 추론 시에만 softmax 적용
        )

    def forward(self, x):
        return self.model(x)

    @property
    def model_name(self): return "Linear (MLP, 수업 방식)"

    @property
    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================
# 데이터셋
# =============================================
def _load_all_samples(root_dir: str):
    samples = []
    for label_idx, cls in enumerate(CLASSES):
        d = os.path.join(root_dir, cls)
        if not os.path.isdir(d):
            print(f"[Dataset] 경고: {d} 없음")
            continue
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith((".jpg", ".png", ".jpeg")):
                samples.append((os.path.join(d, fn), label_idx))
    return samples


def _stratified_split(samples, val_ratio=0.2, seed=42):
    rng = random.Random(seed)
    per_class = {}
    for path, label in samples:
        per_class.setdefault(label, []).append((path, label))
    train_s, val_s = [], []
    for label in sorted(per_class):
        items = per_class[label][:]
        rng.shuffle(items)
        n = max(1, int(len(items) * val_ratio))
        val_s.extend(items[:n])
        train_s.extend(items[n:])
    return train_s, val_s


class VehicleDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])
    return train_tf, val_tf


# =============================================
# 공통 학습 루프 (Gradient Descent)
# =============================================
def _train_one_model(model, train_loader, val_loader, device, save_path):
    """
    Gradient Descent 기반 학습 루프.
    매 에폭마다 loss/acc 기록 후 반환.
    """
    # Multi-class Classification: CrossEntropyLoss (Softmax 내장)
    criterion = nn.CrossEntropyLoss()
    # Gradient Descent 옵티마이저 (AdamW = Adam + Weight Decay)
    optimizer = optim.AdamW(model.parameters(),
                            lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5)

    history = {
        "train_loss": [], "val_acc": [],
        "per_class":  {c: [] for c in CLASSES},
        "best_acc":   0.0,
        "epochs_run": 0,
        "train_sec":  0.0,
    }
    patience_cnt = 0
    t_start      = time.time()

    print(f"\n{'='*70}")
    print(f"  [{model.model_name}]  파라미터: {model.param_count:,}개")
    print(f"{'='*70}")
    print(f"{'Ep':>4}  {'Loss':>8}  {'ValAcc':>8}  "
          + "  ".join(f"{c:>7}" for c in CLASSES) + "  Time")
    print("-" * 70)

    for epoch in range(1, EPOCHS + 1):
        # ── Train (Gradient Descent) ────────────────────────
        model.train()
        run_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()          # 1. Gradient 초기화
            loss = criterion(model(imgs), labels)  # 2. Forward + Loss
            loss.backward()                # 3. Backprop (Gradient 계산)
            optimizer.step()               # 4. 파라미터 업데이트
            run_loss += loss.item()
        scheduler.step()

        # ── Validation ──────────────────────────────────────
        model.eval()
        cls_c = [0] * NUM_CLASSES
        cls_t = [0] * NUM_CLASSES
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(1)
                for c in range(NUM_CLASSES):
                    m = (labels == c)
                    cls_c[c] += (preds[m] == c).sum().item()
                    cls_t[c] += m.sum().item()

        val_acc = sum(cls_c) / max(sum(cls_t), 1)
        per_cls = [cls_c[c] / max(cls_t[c], 1) for c in range(NUM_CLASSES)]
        elapsed = (time.time() - t_start) / 60
        mark    = " ★" if val_acc > history["best_acc"] else ""

        print(f"{epoch:4d}  {run_loss/len(train_loader):8.4f}  "
              f"{val_acc:8.4f}  "
              + "  ".join(f"{a:7.3f}" for a in per_cls)
              + f"  {elapsed:.1f}m{mark}")

        history["train_loss"].append(run_loss / len(train_loader))
        history["val_acc"].append(val_acc)
        for i, c in enumerate(CLASSES):
            history["per_class"][c].append(per_cls[i])

        if val_acc > history["best_acc"]:
            history["best_acc"] = val_acc
            patience_cnt = 0
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"\n[EarlyStopping] {PATIENCE}에폭 개선 없음 → 조기 종료")
                break

        history["epochs_run"] = epoch

    history["train_sec"] = time.time() - t_start

    # ── 히스토리 npy 저장 (regenerate_graphs.py 에서 학습 곡선 재생성 시 사용) ──
    hist_path = save_path.replace(".pth", "_history.npy")
    import numpy as _np
    _np.save(hist_path, history)

    print(f"\n  저장: {save_path}")
    print(f"  히스토리: {hist_path}")
    print(f"  Best Val Acc: {history['best_acc']*100:.2f}%  "
          f"학습시간: {history['train_sec']:.0f}초")
    return history


# =============================================
# 예측 수집 + 추론 속도 측정
# =============================================
def _collect_preds(model, val_loader, device):
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(1)
            all_true.extend(labels.tolist())
            all_pred.extend(preds.cpu().tolist())
    return all_true, all_pred


def _measure_inference_ms(model, device, n=300):
    model.eval()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    for _ in range(20):
        model(dummy)
    t0 = time.time()
    for _ in range(n):
        model(dummy)
    return (time.time() - t0) / n * 1000


# =============================================
# 발표용 그래프 3종 저장
# =============================================
def save_presentation_results(cnn_hist, lin_hist,
                               cnn_true, cnn_pred,
                               lin_true, lin_pred,
                               cnn_ms, lin_ms):
    # 다크 테마 설정
    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 13,
        "figure.facecolor": "#0d1117",
        "axes.facecolor":   "#161b22",
        "text.color": "white", "axes.labelcolor": "white",
        "xtick.color": "white", "ytick.color": "white",
        "axes.edgecolor": "#30363d", "grid.color": "#21262d",
        "legend.facecolor": "#161b22", "legend.edgecolor": "#30363d",
    })

    cnn_acc = accuracy_score(cnn_true, cnn_pred) * 100
    lin_acc = accuracy_score(lin_true, lin_pred) * 100
    cnn_f1  = f1_score(cnn_true, cnn_pred, average="macro") * 100
    lin_f1  = f1_score(lin_true, lin_pred, average="macro") * 100
    cnn_rpt = classification_report(
        cnn_true, cnn_pred, target_names=CLASSES, output_dict=True)
    lin_rpt = classification_report(
        lin_true, lin_pred, target_names=CLASSES, output_dict=True)

    # ─────────────────────────────────────────────────────────
    # ① 학습 곡선 — 4패널
    # ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("학습 결과 비교: CNN vs Linear (수업 모델, Gradient Descent)",
                 fontsize=14, fontweight="bold", color="white", y=0.98)

    ep_cnn = list(range(1, len(cnn_hist["train_loss"]) + 1))
    ep_lin = list(range(1, len(lin_hist["train_loss"]) + 1))

    # (0,0) Train Loss
    ax = axes[0, 0]
    ax.plot(ep_cnn, cnn_hist["train_loss"], "#58a6ff", lw=2, label="CNN")
    ax.plot(ep_lin, lin_hist["train_loss"], "#f78166", lw=2,
            ls="--", label="Linear (수업)")
    ax.set_title("Train Loss (CrossEntropyLoss)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(alpha=0.35)

    # (0,1) Validation Accuracy
    ax = axes[0, 1]
    ax.plot(ep_cnn, [v*100 for v in cnn_hist["val_acc"]],
            "#58a6ff", lw=2,
            label=f"CNN  (best {cnn_hist['best_acc']*100:.1f}%)")
    ax.plot(ep_lin, [v*100 for v in lin_hist["val_acc"]],
            "#f78166", lw=2, ls="--",
            label=f"Linear  (best {lin_hist['best_acc']*100:.1f}%)")
    ax.axhline(cnn_hist["best_acc"]*100, color="#58a6ff", alpha=0.25, ls=":")
    ax.axhline(lin_hist["best_acc"]*100, color="#f78166", alpha=0.25, ls=":")
    ax.set_title("Validation Accuracy")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 108); ax.legend(); ax.grid(alpha=0.35)

    colors_cls = ["#58a6ff", "#56d364", "#ffa657", "#f78166"]

    # (1,0) CNN 클래스별 Acc
    ax = axes[1, 0]
    for i, cls in enumerate(CLASSES):
        vals = [v*100 for v in cnn_hist["per_class"][cls]]
        ax.plot(ep_cnn, vals, color=colors_cls[i], lw=1.5, label=cls.upper())
    ax.set_title("CNN — 클래스별 Validation Accuracy")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 108); ax.legend(fontsize=9); ax.grid(alpha=0.35)

    # (1,1) Linear 클래스별 Acc
    ax = axes[1, 1]
    for i, cls in enumerate(CLASSES):
        vals = [v*100 for v in lin_hist["per_class"][cls]]
        ax.plot(ep_lin, vals, color=colors_cls[i], lw=1.5,
                ls="--", label=cls.upper())
    ax.set_title("Linear — 클래스별 Validation Accuracy")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 108); ax.legend(fontsize=9); ax.grid(alpha=0.35)

    plt.tight_layout()
    out1 = os.path.join(RESULT_DIR, "01_learning_curves.png")
    plt.savefig(out1, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[결과] {out1}")

    # ─────────────────────────────────────────────────────────
    # ② Confusion Matrix (나란히, 행 정규화)
    # ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Confusion Matrix (행=실제 / 열=예측) : CNN vs Linear",
                 fontsize=14, fontweight="bold", color="white")

    for ax, true, pred, title in [
        (axes[0], cnn_true, cnn_pred,
         f"CNN  (Acc {cnn_acc:.1f}%  F1 {cnn_f1:.1f}%)"),
        (axes[1], lin_true, lin_pred,
         f"Linear  (Acc {lin_acc:.1f}%  F1 {lin_f1:.1f}%)"),
    ]:
        cm_raw  = confusion_matrix(true, pred)
        cm_norm = cm_raw.astype(float) / cm_raw.sum(axis=1, keepdims=True)
        sns.heatmap(cm_norm, annot=True, fmt=".2f", ax=ax,
                    xticklabels=[c.upper() for c in CLASSES],
                    yticklabels=[c.upper() for c in CLASSES],
                    cmap="Blues", linewidths=0.5, linecolor="#30363d",
                    annot_kws={"size": 13})
        # 대각선 강조
        for i in range(NUM_CLASSES):
            ax.add_patch(plt.Rectangle(
                (i, i), 1, 1, fill=False,
                edgecolor="#58a6ff", lw=2.5))
        ax.set_title(title, color="white", pad=10)
        ax.set_xlabel("예측 (Predicted)"); ax.set_ylabel("실제 (Actual)")

    plt.tight_layout()
    out2 = os.path.join(RESULT_DIR, "02_confusion_matrix.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[결과] {out2}")

    # ─────────────────────────────────────────────────────────
    # ③ 모델 비교표 — 4패널
    # ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 9), facecolor="#0d1117")
    fig.suptitle("모델 비교 요약: CNN vs Linear (수업 방식, nn.Sequential + GD)",
                 fontsize=14, fontweight="bold", color="white", y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # (0,0) 전체 성능 바 차트
    ax = fig.add_subplot(gs[0, 0])
    ax.set_facecolor("#161b22")
    metrics  = ["Accuracy (%)", "Macro F1 (%)"]
    cnn_vals = [cnn_acc, cnn_f1]
    lin_vals = [lin_acc, lin_f1]
    x = np.arange(len(metrics)); w = 0.32
    b1 = ax.bar(x - w/2, cnn_vals, w, color="#58a6ff", label="CNN",    alpha=0.9)
    b2 = ax.bar(x + w/2, lin_vals, w, color="#f78166", label="Linear", alpha=0.9)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%",
                ha="center", va="bottom", fontsize=11, color="white",
                fontweight="bold")
    ax.set_title("전체 성능 비교 (Accuracy / F1)")
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylim(0, 118); ax.legend(); ax.grid(axis="y", alpha=0.35)

    # (0,1) 수치 비교표
    ax = fig.add_subplot(gs[0, 1])
    ax.set_facecolor("#161b22"); ax.axis("off")

    cnn_params = VehicleCNN().param_count
    lin_params = VehicleLinear().param_count
    rows = [
        ["항목",         "CNN",                       "Linear (수업)"],
        ["구조",         "Conv→BN→ReLU×4 + GAP",      "Flatten→Linear×3"],
        ["파라미터 수",  f"{cnn_params:,}",            f"{lin_params:,}"],
        ["추론 속도",    f"{cnn_ms:.2f} ms/img",       f"{lin_ms:.2f} ms/img"],
        ["학습 시간",    f"{cnn_hist['train_sec']:.0f}초", f"{lin_hist['train_sec']:.0f}초"],
        ["학습 에폭",    str(cnn_hist["epochs_run"]),  str(lin_hist["epochs_run"])],
        ["Best Acc",     f"{cnn_acc:.1f}%",            f"{lin_acc:.1f}%"],
        ["Macro F1",     f"{cnn_f1:.1f}%",             f"{lin_f1:.1f}%"],
        ["Acc 차이",     f"+{cnn_acc-lin_acc:.1f}%p (CNN 우위)", "baseline"],
    ]
    tbl = ax.table(cellText=rows[1:], colLabels=rows[0],
                   cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#30363d")
        cell.set_facecolor("#161b22")
        cell.set_text_props(color="white")
        if r == 0:
            cell.set_facecolor("#21262d")
            cell.set_text_props(color="#58a6ff", fontweight="bold")
        elif c == 1:
            cell.set_facecolor("#0d2240")
        elif c == 2:
            cell.set_facecolor("#200d0d")
    ax.set_title("상세 비교표", color="white", pad=8)

    # (1,0) 클래스별 F1 Score
    ax = fig.add_subplot(gs[1, 0])
    ax.set_facecolor("#161b22")
    cls_f1_cnn = [cnn_rpt[c]["f1-score"]*100 for c in CLASSES]
    cls_f1_lin = [lin_rpt[c]["f1-score"]*100 for c in CLASSES]
    x2 = np.arange(NUM_CLASSES)
    b3 = ax.bar(x2 - w/2, cls_f1_cnn, w, color="#58a6ff",
                label="CNN", alpha=0.9)
    b4 = ax.bar(x2 + w/2, cls_f1_lin, w, color="#f78166",
                label="Linear", alpha=0.9)
    for bar in list(b3) + list(b4):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f"{bar.get_height():.0f}",
                ha="center", va="bottom", fontsize=9, color="white")
    ax.set_title("클래스별 F1 Score (%)")
    ax.set_xticks(x2); ax.set_xticklabels([c.upper() for c in CLASSES])
    ax.set_ylim(0, 118); ax.legend(); ax.grid(axis="y", alpha=0.35)

    # (1,1) Precision 레이더 차트
    ax = fig.add_subplot(gs[1, 1], polar=True)
    ax.set_facecolor("#161b22")
    angles = np.linspace(0, 2*np.pi, NUM_CLASSES, endpoint=False).tolist()
    angles_c = angles + angles[:1]

    cnn_prec = [cnn_rpt[c]["precision"] for c in CLASSES]
    lin_prec = [lin_rpt[c]["precision"] for c in CLASSES]
    cnn_prec_c = cnn_prec + cnn_prec[:1]
    lin_prec_c = lin_prec + lin_prec[:1]

    ax.plot(angles_c, cnn_prec_c, "#58a6ff", lw=2.5, label="CNN Precision")
    ax.fill(angles_c, cnn_prec_c, "#58a6ff", alpha=0.15)
    ax.plot(angles_c, lin_prec_c, "#f78166", lw=2.5,
            ls="--", label="Linear Precision")
    ax.fill(angles_c, lin_prec_c, "#f78166", alpha=0.10)
    ax.set_xticks(angles)
    ax.set_xticklabels([c.upper() for c in CLASSES], color="white", size=11)
    ax.set_ylim(0, 1); ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%","50%","75%","100%"], fontsize=7, color="gray")
    ax.tick_params(colors="gray"); ax.spines["polar"].set_color("#30363d")
    ax.set_title("Precision 레이더 차트", color="white", pad=15)
    ax.legend(loc="lower right", fontsize=8)

    out3 = os.path.join(RESULT_DIR, "03_model_comparison.png")
    plt.savefig(out3, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[결과] {out3}")

    # ── 콘솔 최종 요약 ─────────────────────────────────────────
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  발표용 결과 저장 완료 → {RESULT_DIR}/")
    print(f"{sep}")
    print(f"  {'':20s}  {'CNN':>15}  {'Linear (수업)':>15}")
    print(f"  {'Best Val Accuracy':20s}  {cnn_acc:>14.2f}%  {lin_acc:>14.2f}%")
    print(f"  {'Macro F1':20s}  {cnn_f1:>14.2f}%  {lin_f1:>14.2f}%")
    print(f"  {'파라미터 수':20s}  {cnn_params:>14,}  {lin_params:>14,}")
    print(f"  {'추론 속도':20s}  {cnn_ms:>13.2f}ms  {lin_ms:>13.2f}ms")
    print(f"  {'학습 시간':20s}  {cnn_hist['train_sec']:>13.0f}초  "
          f"{lin_hist['train_sec']:>13.0f}초")
    print(f"{sep}")
    print(f"  CNN 우위: Accuracy +{cnn_acc-lin_acc:.1f}%p  "
          f"F1 +{cnn_f1-lin_f1:.1f}%p")
    print(f"{sep}\n")

    print("[CNN Classification Report]")
    print(classification_report(cnn_true, cnn_pred, target_names=CLASSES))
    print("[Linear Classification Report]")
    print(classification_report(lin_true, lin_pred, target_names=CLASSES))


# =============================================
# 전체 학습 + 비교
# =============================================
def train_all():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Train] Device={device}  IMG={IMG_SIZE}  "
          f"Batch={BATCH_SIZE}  Epochs={EPOCHS}  LR={LEARNING_RATE}")

    all_s = _load_all_samples(DATA_DIR)
    if not all_s:
        raise RuntimeError(f"이미지 없음: {DATA_DIR}")

    train_s, val_s = _stratified_split(all_s)
    train_cnt = Counter(s[1] for s in train_s)
    print("\n[Dataset]")
    for i, c in enumerate(CLASSES):
        print(f"  {c:<6}: train={train_cnt[i]:4d}  "
              f"val={Counter(s[1] for s in val_s)[i]:4d}")

    train_tf, val_tf = get_transforms()
    train_set = VehicleDataset(train_s, train_tf)
    val_set   = VehicleDataset(val_s,   val_tf)

    class_w  = [1.0 / max(train_cnt[i], 1) for i in range(NUM_CLASSES)]
    sample_w = [class_w[s[1]] for s in train_s]
    sampler  = WeightedRandomSampler(sample_w, len(train_s), replacement=True)
    train_ld = DataLoader(train_set, BATCH_SIZE, sampler=sampler,
                          num_workers=0, pin_memory=False)
    val_ld   = DataLoader(val_set,   BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=False)

    # CNN 학습
    cnn_model = VehicleCNN(NUM_CLASSES).to(device)
    cnn_hist  = _train_one_model(cnn_model, train_ld, val_ld, device, CNN_PATH)

    # Linear 학습
    lin_model = VehicleLinear(NUM_CLASSES).to(device)
    lin_hist  = _train_one_model(lin_model, train_ld, val_ld, device, LINEAR_PATH)

    # Best 가중치 로드 후 예측 수집
    cnn_model.load_state_dict(torch.load(CNN_PATH,    map_location=device))
    lin_model.load_state_dict(torch.load(LINEAR_PATH, map_location=device))
    cnn_true, cnn_pred = _collect_preds(cnn_model, val_ld, device)
    lin_true, lin_pred = _collect_preds(lin_model, val_ld, device)

    # 추론 속도
    cnn_ms = _measure_inference_ms(cnn_model, device)
    lin_ms = _measure_inference_ms(lin_model, device)

    save_presentation_results(cnn_hist, lin_hist,
                               cnn_true, cnn_pred,
                               lin_true, lin_pred,
                               cnn_ms, lin_ms)
    return cnn_hist["best_acc"]


# =============================================
# 추론 인터페이스 (monitor에서 호출)
# =============================================
class VehicleClassifier:
    """CNN 모델 추론. classify_with_conf()로 신뢰도 함께 반환."""

    def __init__(self, model_path: str = CNN_PATH):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = VehicleCNN(NUM_CLASSES).to(self.device)
        self.tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ])
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"모델 없음: {model_path}\n"
                "python src/classifier.py 로 먼저 학습하세요.")
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"[Classifier] CNN 로드: {model_path}")

    def classify(self, img: np.ndarray) -> str:
        label, _ = self.classify_with_conf(img)
        return label

    def classify_with_conf(self, img: np.ndarray):
        if img is None or img.size == 0:
            return "car", 0.0
        rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = self.tf(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1)[0]
        conf  = probs.max().item()
        label = CLASSES[probs.argmax().item()]
        return ("uncertain", conf) if conf < CONF_THRESHOLD else (label, conf)


class LinearVehicleClassifier:
    """Linear 모델 추론 (실시간 비교용)."""

    def __init__(self, model_path: str = LINEAR_PATH):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = VehicleLinear(NUM_CLASSES).to(self.device)
        self.tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ])
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"모델 없음: {model_path}\n"
                "python src/classifier.py 로 먼저 학습하세요.")
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"[Classifier] Linear 로드: {model_path}")

    def classify_with_conf(self, img: np.ndarray):
        if img is None or img.size == 0:
            return "car", 0.0
        rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = self.tf(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1)[0]
        conf  = probs.max().item()
        label = CLASSES[probs.argmax().item()]
        return ("uncertain", conf) if conf < CONF_THRESHOLD else (label, conf)


# =============================================
# 단독 실행
# =============================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="차종 분류 학습: CNN vs Linear 비교")
    parser.add_argument("--cnn",    action="store_true", help="CNN만 학습")
    parser.add_argument("--linear", action="store_true", help="Linear만 학습")
    args = parser.parse_args()

    if args.cnn and not args.linear:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        all_s = _load_all_samples(DATA_DIR)
        train_s, val_s = _stratified_split(all_s)
        train_tf, val_tf = get_transforms()
        train_cnt = Counter(s[1] for s in train_s)
        class_w  = [1.0 / max(train_cnt[i], 1) for i in range(NUM_CLASSES)]
        sample_w = [class_w[s[1]] for s in train_s]
        sampler  = WeightedRandomSampler(sample_w, len(train_s), replacement=True)
        train_ld = DataLoader(VehicleDataset(train_s, train_tf),
                              BATCH_SIZE, sampler=sampler, num_workers=0)
        val_ld   = DataLoader(VehicleDataset(val_s, val_tf),
                              BATCH_SIZE, shuffle=False, num_workers=0)
        m = VehicleCNN(NUM_CLASSES).to(device)
        _train_one_model(m, train_ld, val_ld, device, CNN_PATH)
    elif args.linear and not args.cnn:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        all_s = _load_all_samples(DATA_DIR)
        train_s, val_s = _stratified_split(all_s)
        train_tf, val_tf = get_transforms()
        train_cnt = Counter(s[1] for s in train_s)
        class_w  = [1.0 / max(train_cnt[i], 1) for i in range(NUM_CLASSES)]
        sample_w = [class_w[s[1]] for s in train_s]
        sampler  = WeightedRandomSampler(sample_w, len(train_s), replacement=True)
        train_ld = DataLoader(VehicleDataset(train_s, train_tf),
                              BATCH_SIZE, sampler=sampler, num_workers=0)
        val_ld   = DataLoader(VehicleDataset(val_s, val_tf),
                              BATCH_SIZE, shuffle=False, num_workers=0)
        m = VehicleLinear(NUM_CLASSES).to(device)
        _train_one_model(m, train_ld, val_ld, device, LINEAR_PATH)
    else:
        train_all()