# GPU 서버로 옮길 파일 가이드

## 📦 필수 파일 목록

GPU 서버로 옮겨야 할 최소 파일들:

### 1. Docker 관련 파일
```
wsi-rna-loki/
├── Dockerfile                    # 필수: Docker 이미지 빌드용
└── requirements_docker.txt       # 필수: Python 패키지 의존성
```

### 2. 소스 코드
```
wsi-rna-loki/
└── src/
    └── finetune_hnscc.py        # 필수: 파인튜닝 실행 코드
```

### 3. 실행 스크립트
```
wsi-rna-loki/
└── scripts_gpulab/
    └── run_jobs.sh              # 필수: Docker 시작 시 실행되는 스크립트
```

## 🚀 옮기는 방법

### 방법 1: 전체 디렉토리 복사 (권장)
```bash
# 로컬에서
cd /data/hbae
tar -czf wsi-rna-loki.tar.gz wsi-rna-loki/ --exclude='gpulab' --exclude='__pycache__'

# GPU 서버로 전송
scp wsi-rna-loki.tar.gz user@gpu-server:/path/to/destination/

# GPU 서버에서
cd /path/to/destination
tar -xzf wsi-rna-loki.tar.gz
```

### 방법 2: rsync 사용 (더 효율적)
```bash
rsync -avz --exclude='gpulab' --exclude='__pycache__' \
  /data/hbae/wsi-rna-loki/ user@gpu-server:/path/to/destination/wsi-rna-loki/
```

### 방법 3: 필요한 파일만 선택적으로 복사
```bash
# GPU 서버에서 디렉토리 생성
mkdir -p /path/to/destination/wsi-rna-loki/{src,scripts_gpulab}

# 파일 복사
scp Dockerfile user@gpu-server:/path/to/destination/wsi-rna-loki/
scp requirements_docker.txt user@gpu-server:/path/to/destination/wsi-rna-loki/
scp src/finetune_hnscc.py user@gpu-server:/path/to/destination/wsi-rna-loki/src/
scp scripts_gpulab/run_jobs.sh user@gpu-server:/path/to/destination/wsi-rna-loki/scripts_gpulab/
```

## 📋 GPU 서버에서 필요한 추가 사항

### 1. OpenCLIP 레포지토리
```bash
# GPU 서버에 open_clip 레포지토리가 있어야 함
# 예: /data/hbae/open_clip 또는 다른 경로
```

### 2. 체크포인트 파일
```bash
# 사전 학습된 체크포인트 파일
# 예: /data/hbae/checkpoint.pt
```

### 3. 데이터 파일
```bash
# 학습 데이터 CSV 파일
# 예: /data/hbae/Loki_Finetuning/HEG_finetune_meta.csv
```

## 🐳 Docker 이미지 빌드 및 실행

### 1. 이미지 빌드
```bash
cd /path/to/destination/wsi-rna-loki
docker build -t wsi-rna-loki:latest .
```

### 2. 컨테이너 실행
```bash
docker run --gpus all \
  -e OPENCLIP_ROOT=/data/hbae/open_clip \
  -e PRETRAINED=/data/hbae/checkpoint.pt \
  -e CSV_PATH=/data/hbae/Loki_Finetuning/HEG_finetune_meta.csv \
  -v /data/hbae/open_clip:/data/hbae/open_clip \
  -v /data/hbae/checkpoint.pt:/data/hbae/checkpoint.pt \
  -v /data/hbae/Loki_Finetuning:/data/hbae/Loki_Finetuning \
  -v /data/hbae/outputs:/data/hbae/outputs \
  -v /data/hbae/logs:/data/hbae/logs \
  wsi-rna-loki:latest
```

## ✅ 체크리스트

GPU 서버로 옮기기 전 확인:
- [ ] Dockerfile
- [ ] requirements_docker.txt
- [ ] src/finetune_hnscc.py
- [ ] scripts_gpulab/run_jobs.sh

GPU 서버에서 확인:
- [ ] open_clip 레포지토리 존재
- [ ] 체크포인트 파일 존재
- [ ] 데이터 CSV 파일 존재
- [ ] 출력 디렉토리 권한 확인

