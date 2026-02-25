"""
classifier.py
=============
담당: 김세현 (딥러닝 모델링 & 시스템 통합)
브랜치: feature/cnn-model

역할:
    - 경량 CNN 모델 설계 및 학습 (PyTorch)
    - 차종 이진 분류: 승용차(car) vs 트럭(truck)
    - 학습된 모델 저장 및 추론 인터페이스 제공
    - main_system.py에서 호출하는 classify() 함수 구현

산출물:
    - models/classifier.pth (학습된 가중치 - gitignore 처리됨)
    - classify(image_crop) → "car" | "truck" 반환
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

# =============================================
# 설정값 (Config)
# =============================================
MODEL_PATH = "models/classifier.pth"
DATA_DIR = "data/vehicle_images"
IMG_SIZE = 64          # CNN 입력 이미지 크기
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
CLASSES = ["car", "truck"]  # 0: car, 1: truck


# =============================================
# 경량 CNN 모델 정의
# =============================================
class VehicleCNN(nn.Module):
    """
    승용차 / 트럭 이진 분류용 경량 CNN

    입력: (B, 3, 64, 64) RGB 이미지
    출력: (B, 2) 클래스 로짓
    """

    def __init__(self):
        super(VehicleCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1: 3 → 32 채널
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # 64 → 32

            # Block 2: 32 → 64 채널
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # 32 → 16

            # Block 3: 64 → 128 채널
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # 16 → 8
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, len(CLASSES)),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)   # Flatten
        x = self.classifier(x)
        return x


# =============================================
# 데이터셋 클래스
# =============================================
class VehicleDataset(Dataset):
    """
    data/vehicle_images/ 폴더 구조에서 이미지 로드
    폴더 구조:
        data/vehicle_images/
            car/    ← 승용차 이미지들
            truck/  ← 트럭 이미지들
    """

    def __init__(self, root_dir: str, transform=None):
        self.samples = []
        self.transform = transform

        for label_idx, class_name in enumerate(CLASSES):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"[Dataset] 경고: {class_dir} 폴더가 없습니다.")
                continue
            for filename in os.listdir(class_dir):
                if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                    self.samples.append(
                        (os.path.join(class_dir, filename), label_idx)
                    )

        print(f"[Dataset] 총 {len(self.samples)}개 샘플 로드")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# =============================================
# 학습 파이프라인
# =============================================
def get_transforms():
    """학습/추론용 전처리 변환 정의"""
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform


def train_model():
    """CNN 모델 학습 및 가중치 저장"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Classifier] 학습 시작 - Device: {device}")

    train_transform, val_transform = get_transforms()
    dataset = VehicleDataset(DATA_DIR, transform=train_transform)

    # 학습/검증 분할 (8:2)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    val_set.dataset.transform = val_transform

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    model = VehicleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        # 학습
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # 검증
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total
        scheduler.step()

        print(
            f"[Classifier] Epoch {epoch+1}/{EPOCHS} | "
            f"Loss: {running_loss/len(train_loader):.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        # 최고 성능 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"[Classifier] 모델 저장 완료: {MODEL_PATH} (Acc: {val_acc:.4f})")

    print(f"[Classifier] 학습 완료. 최고 검증 정확도: {best_val_acc:.4f}")


# =============================================
# 추론 인터페이스 (main_system.py에서 호출)
# =============================================
class VehicleClassifier:
    """
    학습된 CNN 모델을 로드하고 추론을 수행하는 클래스
    main_system.py에서 인스턴스화하여 사용합니다.
    """

    def __init__(self, model_path: str = MODEL_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VehicleCNN().to(self.device)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self._load_model(model_path)

    def _load_model(self, model_path: str):
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
        차량 크롭 이미지를 받아 차종 반환
        (vision_processor.py의 DetectedObject.bbox 기반 크롭 이미지를 입력)

        Args:
            image_crop: BGR numpy 배열 (차량 영역 크롭)

        Returns:
            "car" 또는 "truck"
        """
        if image_crop is None or image_crop.size == 0:
            return "car"  # 기본값

        # BGR → RGB 변환 후 전처리
        rgb_crop = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb_crop).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            _, predicted = torch.max(outputs, 1)

        return CLASSES[predicted.item()]


if __name__ == "__main__":
    # 단독 실행 시 학습 모드
    train_model()
