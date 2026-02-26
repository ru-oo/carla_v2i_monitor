"""
classifier.py
=============
담당: 김세현 (딥러닝 모델링 & 시스템 통합)
브랜치: feature/cnn-model

역할:
    - 경량 CNN 모델 설계 및 학습 (PyTorch)
    - 차종 4클래스 분류: bus / car / truck / van
    - 학습된 모델 저장 및 추론 인터페이스 제공
    - main_system.py에서 호출하는 classify() 함수 구현

산출물:
    - models/classifier.pth (학습된 가중치 - gitignore 처리됨)
    - classify(image_crop) → "bus" | "car" | "truck" | "van" 반환
"""

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

# =============================================
# 경로 설정 (스크립트 위치 기준 — CWD 무관)
# =============================================
_SRC_DIR   = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.dirname(_SRC_DIR)
DATA_DIR   = os.path.join(BASE_DIR, "data", "vehicle_images")
MODEL_PATH = os.path.join(BASE_DIR, "models", "classifier.pth")

# =============================================
# 설정값 (Config)
# =============================================
IMG_SIZE      = 64          # CNN 입력 크기 (CPU 친화적)
BATCH_SIZE    = 64
EPOCHS        = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
PATIENCE      = 7           # Early-stopping patience

CLASSES     = ["bus", "car", "truck", "van"]   # 4클래스
NUM_CLASSES = len(CLASSES)


# =============================================
# CNN 모델 정의 (4-Block + GlobalAvgPool)
# =============================================
class VehicleCNN(nn.Module):
    """
    4클래스 차종 분류 CNN

    아키텍처:
        Block 1~4: Conv 3×3 (×2) + BN + ReLU + MaxPool2×2
        GAP (Global Average Pooling) → 파라미터 절감
        FC: 256 → 128 → NUM_CLASSES

    입력: (B, 3, 64, 64)
    출력: (B, NUM_CLASSES)
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),        # 해상도 ÷2
                nn.Dropout2d(0.1),
            )

        self.features = nn.Sequential(
            _conv_block(3,    32),    # 64 → 32
            _conv_block(32,   64),    # 32 → 16
            _conv_block(64,  128),    # 16 →  8
            _conv_block(128, 256),    #  8 →  4
        )

        # Global Average Pooling: (B, 256, 4, 4) → (B, 256)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        x = self.head(x)
        return x


# =============================================
# 데이터셋 클래스
# =============================================
def _load_all_samples(root_dir: str) -> list[tuple[str, int]]:
    """
    root_dir 아래 CLASSES 폴더를 스캔해
    (절대경로, label_idx) 리스트 반환
    """
    samples: list[tuple[str, int]] = []
    for label_idx, class_name in enumerate(CLASSES):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"[Dataset] 경고: {class_dir} 없음 — 스킵")
            continue
        for fn in sorted(os.listdir(class_dir)):
            if fn.lower().endswith((".jpg", ".png", ".jpeg")):
                samples.append((os.path.join(class_dir, fn), label_idx))
    return samples


class VehicleDataset(Dataset):
    """(path, label) 리스트를 받아 이미지를 반환하는 Dataset"""

    def __init__(self, samples: list[tuple[str, int]], transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# =============================================
# Stratified 분할 (클래스별 8:2)
# =============================================
def _stratified_split(
    samples: list[tuple[str, int]],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list, list]:
    """
    각 클래스에서 독립적으로 val_ratio 비율을 검증셋으로 분리.
    Train/Val 간 transform 오염 없이 별도 Dataset 인스턴스로 분리.
    """
    rng = random.Random(seed)
    per_class: dict[int, list] = {}
    for path, label in samples:
        per_class.setdefault(label, []).append((path, label))

    train_s, val_s = [], []
    for label in sorted(per_class.keys()):
        items = per_class[label][:]
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_ratio))
        val_s.extend(items[:n_val])
        train_s.extend(items[n_val:])

    return train_s, val_s


# =============================================
# 전처리 변환
# =============================================
def get_transforms():
    """Train / Val 전처리 변환 반환"""
    _MEAN = [0.485, 0.456, 0.406]
    _STD  = [0.229, 0.224, 0.225]

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
# 학습 파이프라인
# =============================================
def train_model() -> float:
    """
    CNN 모델 학습, 최고 Val Accuracy 모델 저장,
    최종 Confusion Matrix 출력 후 best_val_acc 반환.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"[Classifier] 학습 시작  Device={device}  IMG={IMG_SIZE}  Epochs={EPOCHS}")
    print(f"{'='*60}")

    # ---- 데이터 로드 및 분할 ----
    all_samples = _load_all_samples(DATA_DIR)
    if not all_samples:
        raise RuntimeError(f"이미지를 찾을 수 없습니다: {DATA_DIR}")

    train_samples, val_samples = _stratified_split(all_samples)

    # 분포 출력
    train_cnt = Counter(s[1] for s in train_samples)
    val_cnt   = Counter(s[1] for s in val_samples)
    print("[Dataset] 클래스별 샘플 수:")
    for i, cn in enumerate(CLASSES):
        print(f"  {cn:<6}: train={train_cnt[i]:4d}, val={val_cnt[i]:4d}")
    print(f"  합계 : train={len(train_samples)},  val={len(val_samples)}")

    # ---- DataLoader (train: WeightedSampler, val: 순서 고정) ----
    train_tf, val_tf = get_transforms()
    train_set = VehicleDataset(train_samples, train_tf)
    val_set   = VehicleDataset(val_samples,   val_tf)

    # 클래스 불균형 보정 — WeightedRandomSampler
    class_counts  = [max(train_cnt[i], 1) for i in range(NUM_CLASSES)]
    class_w       = [1.0 / c for c in class_counts]
    sample_w      = [class_w[s[1]] for s in train_samples]
    sampler       = WeightedRandomSampler(sample_w, len(train_samples), replacement=True)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=False)

    # ---- 모델 / 옵티마이저 ----
    model     = VehicleCNN(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    best_val_acc   = 0.0
    patience_count = 0
    t0             = time.time()

    print(f"\n{'Ep':>4}  {'Loss':>7}  {'ValAcc':>7}  "
          + "  ".join(f"{c:>6}" for c in CLASSES)
          + "  Time")
    print("-" * 70)

    for epoch in range(1, EPOCHS + 1):
        # ---- 학습 ----
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()

        # ---- 검증 ----
        model.eval()
        cls_correct = [0] * NUM_CLASSES
        cls_total   = [0] * NUM_CLASSES
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                preds = model(images).argmax(dim=1)
                for c in range(NUM_CLASSES):
                    mask = (labels == c)
                    cls_correct[c] += (preds[mask] == c).sum().item()
                    cls_total[c]   += mask.sum().item()

        val_acc  = sum(cls_correct) / max(sum(cls_total), 1)
        per_cls  = "  ".join(
            f"{cls_correct[c]/max(cls_total[c],1):6.3f}"
            for c in range(NUM_CLASSES)
        )
        elapsed  = (time.time() - t0) / 60
        mark     = " ★" if val_acc > best_val_acc else ""
        print(f"{epoch:4d}  {running_loss/len(train_loader):7.4f}  "
              f"{val_acc:7.4f}  {per_cls}  {elapsed:4.1f}m{mark}")

        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            patience_count = 0
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"\n[EarlyStopping] {PATIENCE} 에폭 개선 없음 — 학습 조기 종료")
                break

    # ---- Confusion Matrix ----
    print(f"\n{'='*60}")
    print(f"[Classifier] 학습 완료  Best Val Acc: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"{'='*60}")

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    confusion = [[0] * NUM_CLASSES for _ in range(NUM_CLASSES)]
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            for t, p in zip(labels.tolist(), preds.tolist()):
                confusion[t][p] += 1

    print("\n[Confusion Matrix]  행=실제, 열=예측")
    header = "       " + "  ".join(f"{c:>6}" for c in CLASSES)
    print(header)
    print("       " + "  ".join(["------"] * NUM_CLASSES))
    for i in range(NUM_CLASSES):
        row_str = "  ".join(f"{confusion[i][c]:6d}" for c in range(NUM_CLASSES))
        print(f"{CLASSES[i]:>6} | {row_str}")

    print(f"\n클래스별 정밀도 (Precision):")
    for c in range(NUM_CLASSES):
        col_sum = sum(confusion[r][c] for r in range(NUM_CLASSES))
        prec = confusion[c][c] / max(col_sum, 1)
        rec  = confusion[c][c] / max(sum(confusion[c]), 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-9)
        print(f"  {CLASSES[c]:<6}: Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}")

    print(f"\n최종 Val Accuracy : {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"모델 저장 위치     : {MODEL_PATH}")
    return best_val_acc


# =============================================
# 추론 인터페이스 (main_system.py에서 호출)
# =============================================
class VehicleClassifier:
    """
    학습된 CNN 모델을 로드하고 추론을 수행하는 클래스.
    main_system.py에서 인스턴스화하여 사용합니다.
    """

    def __init__(self, model_path: str = MODEL_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = VehicleCNN(NUM_CLASSES).to(self.device)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        self._load_model(model_path)

    def _load_model(self, model_path: str) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"모델 파일 없음: {model_path}\n"
                "먼저 python src/classifier.py 로 학습을 실행하세요."
            )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"[Classifier] 모델 로드 완료: {model_path}")

    def classify(self, image_crop: np.ndarray) -> str:
        """
        차량 크롭 이미지(BGR numpy)를 받아 차종 문자열 반환.

        Args:
            image_crop: BGR numpy 배열 (차량 영역 크롭)

        Returns:
            "bus" | "car" | "truck" | "van"
        """
        if image_crop is None or image_crop.size == 0:
            return "car"   # 기본값

        rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred_idx = self.model(tensor).argmax(dim=1).item()
        return CLASSES[pred_idx]


# =============================================
# 단독 실행 시 학습 모드
# =============================================
if __name__ == "__main__":
    train_model()
