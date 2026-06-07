"""
regenerate_graphs.py  — 그래프만 다시 생성 (재학습 없음)
===========================================================
저장된 모델 가중치(.pth)를 불러와 검증셋 예측만 다시 수행,
한글 폰트를 올바르게 설정한 뒤 PNG 3장을 results/ 에 저장합니다.

실행:
    python src/regenerate_graphs.py
"""

import os, sys, time, random
from collections import Counter
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ── 한글 폰트 설정 (matplotlib import 전에 적용) ────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm

_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",   # Linux
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",           # Linux(nanum)
    "C:/Windows/Fonts/malgun.ttf",                                # Windows
    "/Library/Fonts/AppleGothic.ttf",                            # macOS
]
_KOREAN_FONT = next((p for p in _FONT_CANDIDATES if os.path.exists(p)), None)

if _KOREAN_FONT:
    _fe = fm.FontEntry(fname=_KOREAN_FONT, name="KoreanFont")
    fm.fontManager.ttflist.insert(0, _fe)
    matplotlib.rcParams["font.family"] = "KoreanFont"
    print(f"[폰트] {os.path.basename(_KOREAN_FONT)}")
else:
    print("[경고] 한글 폰트 없음 — 영문 표시")

matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (accuracy_score, f1_score,
                              classification_report, confusion_matrix)

# 다크 테마
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13,
    "figure.facecolor": "#0d1117", "axes.facecolor": "#161b22",
    "text.color": "white", "axes.labelcolor": "white",
    "xtick.color": "white", "ytick.color": "white",
    "axes.edgecolor": "#30363d", "grid.color": "#21262d",
    "legend.facecolor": "#161b22", "legend.edgecolor": "#30363d",
})

# ── 경로 설정 ────────────────────────────────────────────────
_SRC_DIR   = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.dirname(_SRC_DIR)
DATA_DIR   = os.path.join(BASE_DIR, "data", "vehicle_images")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

CNN_PATH    = os.path.join(MODEL_DIR, "classifier_cnn.pth")
LINEAR_PATH = os.path.join(MODEL_DIR, "classifier_linear.pth")

sys.path.insert(0, _SRC_DIR)
from classifier import (VehicleCNN, VehicleLinear,
                        CLASSES, NUM_CLASSES, IMG_SIZE, _MEAN, _STD,
                        _load_all_samples, _stratified_split, VehicleDataset)


# =============================================
# 유틸
# =============================================
def collect_preds(model, val_loader, device):
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            preds = model(imgs.to(device)).argmax(1)
            all_true.extend(labels.tolist())
            all_pred.extend(preds.cpu().tolist())
    return all_true, all_pred


def measure_ms(model, device, n=300):
    model.eval()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    for _ in range(20): model(dummy)
    t0 = time.time()
    for _ in range(n): model(dummy)
    return (time.time() - t0) / n * 1000


def load_history(pth_path):
    npy = pth_path.replace(".pth", "_history.npy")
    if os.path.exists(npy):
        return np.load(npy, allow_pickle=True).item()
    return None


# =============================================
# 그래프 ① 학습 곡선
# =============================================
def save_learning_curves(cnn_hist, lin_hist):
    if cnn_hist is None and lin_hist is None:
        print("[스킵] _history.npy 파일 없음 — 학습 곡선 생략")
        print("       (classifier.py 재실행 시 자동 저장됩니다)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("학습 결과 비교: CNN vs Linear (수업 모델, Gradient Descent)",
                 fontsize=14, fontweight="bold", color="white", y=0.98)

    ep_c = list(range(1, len(cnn_hist["train_loss"])+1)) if cnn_hist else []
    ep_l = list(range(1, len(lin_hist["train_loss"])+1)) if lin_hist else []
    cls_colors = ["#58a6ff", "#56d364", "#ffa657", "#f78166"]

    ax = axes[0, 0]
    if cnn_hist: ax.plot(ep_c, cnn_hist["train_loss"], "#58a6ff", lw=2, label="CNN")
    if lin_hist: ax.plot(ep_l, lin_hist["train_loss"], "#f78166", lw=2, ls="--", label="Linear (수업)")
    ax.set_title("Train Loss (CrossEntropyLoss)"); ax.set_xlabel("에폭"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(alpha=0.35)

    ax = axes[0, 1]
    if cnn_hist:
        ax.plot(ep_c, [v*100 for v in cnn_hist["val_acc"]], "#58a6ff", lw=2,
                label=f"CNN (최고 {cnn_hist['best_acc']*100:.1f}%)")
    if lin_hist:
        ax.plot(ep_l, [v*100 for v in lin_hist["val_acc"]], "#f78166", lw=2, ls="--",
                label=f"Linear (최고 {lin_hist['best_acc']*100:.1f}%)")
    ax.set_title("검증 정확도 (Validation Accuracy)"); ax.set_xlabel("에폭"); ax.set_ylabel("정확도 (%)")
    ax.set_ylim(0, 108); ax.legend(); ax.grid(alpha=0.35)

    ax = axes[1, 0]
    if cnn_hist:
        for i, cls in enumerate(CLASSES):
            ax.plot(ep_c, [v*100 for v in cnn_hist["per_class"][cls]],
                    color=cls_colors[i], lw=1.5, label=cls.upper())
    ax.set_title("CNN — 클래스별 검증 정확도"); ax.set_xlabel("에폭"); ax.set_ylabel("정확도 (%)")
    ax.set_ylim(0, 108); ax.legend(fontsize=9); ax.grid(alpha=0.35)

    ax = axes[1, 1]
    if lin_hist:
        for i, cls in enumerate(CLASSES):
            ax.plot(ep_l, [v*100 for v in lin_hist["per_class"][cls]],
                    color=cls_colors[i], lw=1.5, ls="--", label=cls.upper())
    ax.set_title("Linear — 클래스별 검증 정확도"); ax.set_xlabel("에폭"); ax.set_ylabel("정확도 (%)")
    ax.set_ylim(0, 108); ax.legend(fontsize=9); ax.grid(alpha=0.35)

    plt.tight_layout()
    out = os.path.join(RESULT_DIR, "01_learning_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[저장] {out}")


# =============================================
# 그래프 ② Confusion Matrix
# =============================================
def save_confusion_matrix(cnn_true, cnn_pred, lin_true, lin_pred,
                           cnn_acc, lin_acc, cnn_f1, lin_f1):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("혼동 행렬 (Confusion Matrix) : CNN vs Linear",
                 fontsize=14, fontweight="bold", color="white")

    for ax, true, pred, title in [
        (axes[0], cnn_true, cnn_pred, f"CNN  (정확도 {cnn_acc:.1f}%  F1 {cnn_f1:.1f}%)"),
        (axes[1], lin_true, lin_pred, f"Linear  (정확도 {lin_acc:.1f}%  F1 {lin_f1:.1f}%)"),
    ]:
        cm_raw  = confusion_matrix(true, pred)
        cm_norm = cm_raw.astype(float) / cm_raw.sum(axis=1, keepdims=True)
        sns.heatmap(cm_norm, annot=True, fmt=".2f", ax=ax,
                    xticklabels=[c.upper() for c in CLASSES],
                    yticklabels=[c.upper() for c in CLASSES],
                    cmap="Blues", linewidths=0.5, linecolor="#30363d",
                    annot_kws={"size": 13})
        for i in range(NUM_CLASSES):
            ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False,
                                       edgecolor="#58a6ff", lw=2.5))
        ax.set_title(title, color="white", pad=10)
        ax.set_xlabel("예측 (Predicted)"); ax.set_ylabel("실제 (Actual)")

    plt.tight_layout()
    out = os.path.join(RESULT_DIR, "02_confusion_matrix.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[저장] {out}")


# =============================================
# 그래프 ③ 모델 비교표
# =============================================
def save_model_comparison(cnn_true, cnn_pred, lin_true, lin_pred,
                           cnn_acc, lin_acc, cnn_f1, lin_f1,
                           cnn_ms, lin_ms):
    cnn_rpt = classification_report(cnn_true, cnn_pred, target_names=CLASSES, output_dict=True)
    lin_rpt = classification_report(lin_true, lin_pred, target_names=CLASSES, output_dict=True)
    cnn_params = VehicleCNN().param_count
    lin_params = VehicleLinear().param_count

    fig = plt.figure(figsize=(14, 9), facecolor="#0d1117")
    fig.suptitle("모델 비교 요약: CNN vs Linear (수업 방식, nn.Sequential + GD)",
                 fontsize=14, fontweight="bold", color="white", y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # (0,0) 전체 성능 바 차트
    ax = fig.add_subplot(gs[0, 0]); ax.set_facecolor("#161b22")
    x = np.arange(2); w = 0.32
    b1 = ax.bar(x-w/2, [cnn_acc, cnn_f1], w, color="#58a6ff", label="CNN", alpha=0.9)
    b2 = ax.bar(x+w/2, [lin_acc, lin_f1], w, color="#f78166", label="Linear", alpha=0.9)
    for bar in list(b1)+list(b2):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom",
                fontsize=11, color="white", fontweight="bold")
    ax.set_title("전체 성능 비교"); ax.set_xticks(x)
    ax.set_xticklabels(["정확도 (%)", "매크로 F1 (%)"])
    ax.set_ylim(0, 118); ax.legend(); ax.grid(axis="y", alpha=0.35)

    # (0,1) 수치 비교표
    ax = fig.add_subplot(gs[0, 1]); ax.set_facecolor("#161b22"); ax.axis("off")
    rows = [
        ["항목",         "CNN",                       "Linear (수업)"],
        ["구조",         "합성곱(×4)+GAP",            "Flatten→전결합(×3)"],
        ["파라미터 수",  f"{cnn_params:,}",            f"{lin_params:,}"],
        ["추론 속도",    f"{cnn_ms:.2f}ms/img",        f"{lin_ms:.2f}ms/img"],
        ["검증 정확도",  f"{cnn_acc:.1f}%",            f"{lin_acc:.1f}%"],
        ["매크로 F1",    f"{cnn_f1:.1f}%",             f"{lin_f1:.1f}%"],
        ["정확도 차이",  f"+{cnn_acc-lin_acc:.1f}%p↑", "기준 모델"],
    ]
    tbl = ax.table(cellText=rows[1:], colLabels=rows[0],
                   cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#30363d"); cell.set_facecolor("#161b22")
        cell.set_text_props(color="white")
        if r == 0:
            cell.set_facecolor("#21262d")
            cell.set_text_props(color="#58a6ff", fontweight="bold")
        elif c == 1: cell.set_facecolor("#0d2240")
        elif c == 2: cell.set_facecolor("#200d0d")
    ax.set_title("상세 비교표", color="white", pad=8)

    # (1,0) 클래스별 F1 바 차트
    ax = fig.add_subplot(gs[1, 0]); ax.set_facecolor("#161b22")
    x2 = np.arange(NUM_CLASSES)
    cls_f1_cnn = [cnn_rpt[c]["f1-score"]*100 for c in CLASSES]
    cls_f1_lin = [lin_rpt[c]["f1-score"]*100 for c in CLASSES]
    b3 = ax.bar(x2-w/2, cls_f1_cnn, w, color="#58a6ff", label="CNN", alpha=0.9)
    b4 = ax.bar(x2+w/2, cls_f1_lin, w, color="#f78166", label="Linear", alpha=0.9)
    for bar in list(b3)+list(b4):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=9, color="white")
    ax.set_title("클래스별 F1 Score (%)"); ax.set_xticks(x2)
    ax.set_xticklabels([c.upper() for c in CLASSES])
    ax.set_ylim(0, 118); ax.legend(); ax.grid(axis="y", alpha=0.35)

    # (1,1) Precision 레이더 차트
    ax = fig.add_subplot(gs[1, 1], polar=True); ax.set_facecolor("#161b22")
    angles = np.linspace(0, 2*np.pi, NUM_CLASSES, endpoint=False).tolist()
    ac = angles + angles[:1]
    cp = [cnn_rpt[c]["precision"] for c in CLASSES] + [cnn_rpt[CLASSES[0]]["precision"]]
    lp = [lin_rpt[c]["precision"] for c in CLASSES] + [lin_rpt[CLASSES[0]]["precision"]]
    ax.plot(ac, cp, "#58a6ff", lw=2.5, label="CNN Precision")
    ax.fill(ac, cp, "#58a6ff", alpha=0.15)
    ax.plot(ac, lp, "#f78166", lw=2.5, ls="--", label="Linear Precision")
    ax.fill(ac, lp, "#f78166", alpha=0.10)
    ax.set_xticks(angles); ax.set_xticklabels([c.upper() for c in CLASSES], color="white", size=11)
    ax.set_ylim(0, 1); ax.set_yticks([0.25,0.5,0.75,1.0])
    ax.set_yticklabels(["25%","50%","75%","100%"], fontsize=7, color="gray")
    ax.tick_params(colors="gray"); ax.spines["polar"].set_color("#30363d")
    ax.set_title("Precision 레이더 차트", color="white", pad=15)
    ax.legend(loc="lower right", fontsize=8)

    out = os.path.join(RESULT_DIR, "03_model_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[저장] {out}")


# =============================================
# 메인
# =============================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[장치] {device}")

    missing = [p for p in (CNN_PATH, LINEAR_PATH) if not os.path.exists(p)]
    if missing:
        print("\n[오류] 모델 파일 없음:")
        for p in missing: print(f"  {p}")
        print("\n학습 먼저 실행: python src/classifier.py")
        return

    print("\n[데이터] 검증셋 로딩...")
    all_s = _load_all_samples(DATA_DIR)
    if not all_s:
        print(f"[오류] 이미지 없음: {DATA_DIR}"); return
    _, val_s = _stratified_split(all_s)
    val_cnt  = Counter(s[1] for s in val_s)
    for i, c in enumerate(CLASSES):
        print(f"  {c}: {val_cnt[i]}장")

    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])
    val_loader = DataLoader(VehicleDataset(val_s, val_tf),
                            batch_size=64, shuffle=False, num_workers=0)

    print("\n[예측] CNN ...")
    cnn_m = VehicleCNN(NUM_CLASSES).to(device)
    cnn_m.load_state_dict(torch.load(CNN_PATH, map_location=device))
    cnn_true, cnn_pred = collect_preds(cnn_m, val_loader, device)

    print("[예측] Linear ...")
    lin_m = VehicleLinear(NUM_CLASSES).to(device)
    lin_m.load_state_dict(torch.load(LINEAR_PATH, map_location=device))
    lin_true, lin_pred = collect_preds(lin_m, val_loader, device)

    cnn_acc = accuracy_score(cnn_true, cnn_pred) * 100
    lin_acc = accuracy_score(lin_true, lin_pred) * 100
    cnn_f1  = f1_score(cnn_true, cnn_pred, average="macro") * 100
    lin_f1  = f1_score(lin_true, lin_pred, average="macro") * 100
    cnn_ms  = measure_ms(cnn_m, device)
    lin_ms  = measure_ms(lin_m, device)

    cnn_hist = load_history(CNN_PATH)
    lin_hist = load_history(LINEAR_PATH)

    print("\n[그래프] 생성 중 ...")
    save_learning_curves(cnn_hist, lin_hist)
    save_confusion_matrix(cnn_true, cnn_pred, lin_true, lin_pred,
                          cnn_acc, lin_acc, cnn_f1, lin_f1)
    save_model_comparison(cnn_true, cnn_pred, lin_true, lin_pred,
                          cnn_acc, lin_acc, cnn_f1, lin_f1, cnn_ms, lin_ms)

    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  {'':16s}  {'CNN':>14}  {'Linear (수업)':>14}")
    print(f"  {'검증 정확도':16s}  {cnn_acc:>13.2f}%  {lin_acc:>13.2f}%")
    print(f"  {'매크로 F1':16s}  {cnn_f1:>13.2f}%  {lin_f1:>13.2f}%")
    print(f"  {'파라미터 수':16s}  {VehicleCNN().param_count:>14,}  {VehicleLinear().param_count:>14,}")
    print(f"  {'추론 속도':16s}  {cnn_ms:>12.2f}ms  {lin_ms:>12.2f}ms")
    print(f"{sep}")
    print(f"  CNN 우위: 정확도 +{cnn_acc-lin_acc:.1f}%p  F1 +{cnn_f1-lin_f1:.1f}%p")
    print(f"  결과 → {RESULT_DIR}/")
    print(f"{sep}\n")

    print("[CNN Classification Report]")
    print(classification_report(cnn_true, cnn_pred, target_names=CLASSES))
    print("[Linear Classification Report]")
    print(classification_report(lin_true, lin_pred, target_names=CLASSES))


if __name__ == "__main__":
    main()