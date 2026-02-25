# CARLA 기반 V2I 스마트 교차로 3D 관제 시스템 (No-YOLO)

> KDT DL & OpenCV 프로젝트 | CARLA + OpenCV + PyTorch

---

## 프로젝트 개요

YOLO 없이 **수학적 컴퓨터 비전 원리**와 **경량 CNN**만을 활용하여,
고정 CCTV 카메라 기반 V2I(Vehicle-to-Infrastructure) 교차로 3D 관제 시스템 구현

**핵심 파이프라인:**
```
CARLA CCTV 영상 → MOG2 배경차분 → IPM 좌표 변환 → CNN 차종 분류 → 3D 레이더 맵
```

---

## 팀원 역할 분담

| 팀원 | 역할 | 담당 브랜치 | 핵심 파일 |
|------|------|------------|-----------|
| 김예진 | CARLA 시뮬레이션 & 데이터 엔지니어링 | `feature/carla-sim` | `src/data_collector.py` |
| 김진수 | 컴퓨터 비전 (OpenCV) 알고리즘 | `feature/vision-logic` | `src/vision_processor.py` |
| 김세현 | 딥러닝 모델링 & 시스템 통합 | `feature/cnn-model` | `src/classifier.py`, `src/main_system.py` |

---

## 디렉토리 구조

```
carla_v2i_monitor/
├── src/
│   ├── data_collector.py    # [김예진] CARLA 환경 구축 & 데이터 수집
│   ├── vision_processor.py  # [김진수] MOG2 + IPM 좌표 변환
│   ├── classifier.py        # [김세현] CNN 학습 & 차종 분류
│   └── main_system.py       # [전체] 최종 통합 관제 시스템
├── data/
│   └── vehicle_images/      # 학습 데이터 (gitignore - 업로드 금지!)
│       ├── car/
│       └── truck/
├── models/                  # 모델 가중치 (gitignore - 업로드 금지!)
├── logs/
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 기술 스택

| 기술 | 버전 | 용도 |
|------|------|------|
| CARLA Simulator | 0.9.16 / 0.10.0 | 시뮬레이션 환경 |
| Python | 3.10 | 전체 개발 언어 |
| OpenCV | 4.10.0+ | MOG2 배경차분, IPM 변환 |
| PyTorch | 2.0.0+ | CNN 모델 학습 & 추론 |
| NumPy | 1.24.0+ | 행렬 연산 |

---

## 환경 설정

### 1. Python 가상환경 생성
```bash
python3.10 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 2. 패키지 설치
```bash
pip install -r requirements.txt
```

### 3. CARLA Python API 경로 설정
```bash
# CARLA 설치 경로에 맞게 수정
export PYTHONPATH=$PYTHONPATH:/opt/carla-simulator/PythonAPI/carla/dist/carla-0.9.16-py3.10-linux-x86_64.egg
```

### 4. CARLA 서버 실행
```bash
./CarlaUE4.sh -RenderOffScreen   # CARLA 서버 실행
```

---

## 실행 방법

### Step 1: 학습 데이터 수집 (김예진)
```bash
python src/data_collector.py
# data/vehicle_images/ 에 차종별 이미지 저장
```

### Step 2: CNN 모델 학습 (김세현)
```bash
python src/classifier.py
# models/classifier.pth 생성
```

### Step 3: 관제 시스템 실행 (전체)
```bash
python src/main_system.py
# CARLA 서버가 실행 중이어야 합니다
```

---

## Git 협업 규칙

### 브랜치 전략
```
main                  ← 최종 통합 (직접 작업 금지!)
├── feature/carla-sim    ← 김예진 작업 공간
├── feature/vision-logic ← 김진수 작업 공간
└── feature/cnn-model    ← 김세현 작업 공간
```

### 커밋 메시지 규칙
```
feat: 새로운 기능 추가
fix: 버그 수정
docs: 문서 수정
refactor: 코드 리팩토링
test: 테스트 코드

예시:
feat: MOG2 배경차분 노이즈 제거 로직 추가
fix: IPM 변환 행렬 좌표 오류 수정
```

### Pull Request 규칙
1. `feature/*` 브랜치에서 작업 후 `main`으로 PR 생성
2. **최소 1명 이상의 팀원 리뷰(Approve)** 후 병합
3. PR 제목: `[역할] 작업 내용` 형식 사용

### 주의사항
- `.ipynb` 파일 금지 (반드시 `.py`로 변환)
- `data/vehicle_images/`, `models/*.pth` 업로드 금지 (`.gitignore` 처리됨)
- `main` 브랜치에 직접 push 금지

---

## 데이터 처리 흐름

```
[김예진] CARLA에서 영상 획득 (Input Frame)
    ↓
[김진수] MOG2로 움직이는 차량 감지 → 좌표(ROI) 추출
    ↓
[김세현] ROI 크롭 이미지 → CNN 차종 분류 (car / truck)
    ↓
[전체] 3D 레이더 맵 + 원본 영상 오버레이 렌더링
```
