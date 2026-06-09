#!/bin/bash

# Fine-tuning 자동 실행 스크립트
# Docker 컨테이너 시작 시 자동으로 실행됨

set -e  # 에러 발생 시 스크립트 중단

echo "=========================================="
echo "Fine-tuning 작업 시작"
echo "=========================================="
echo "시작 시간: $(date)"
echo "=========================================="

# 기본 경로 설정 (환경 변수로 오버라이드 가능)
OPENCLIP_ROOT="${OPENCLIP_ROOT:-/data/hbae/open_clip}"
PRETRAINED="${PRETRAINED:-/data/hbae/checkpoint.pt}"
CSV_PATH="${CSV_PATH:-/data/hbae/Loki_Finetuning/HEG_finetune_meta.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/hbae/outputs}"
LOG_DIR="${LOG_DIR:-/data/hbae/logs}"

# GPU 설정
GPU_NUM="${GPU_NUM:-0}"
export CUDA_VISIBLE_DEVICES=$GPU_NUM

# WandB 설정
export WANDB_API_KEY="${WANDB_API_KEY:-a8d69bdb4fd712911b58b798d2045e86cd34a4d0}"

# 하이퍼파라미터 (환경 변수로 오버라이드 가능)
export MODEL_NAME="${MODEL_NAME:-coca_ViT-L-14}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-finetune_HEG_hnscc}"
export EPOCHS="${EPOCHS:-5}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export LR="${LR:-5e-6}"
export WD="${WD:-0.1}"
export WARMUP="${WARMUP:-10}"
export WORKERS="${WORKERS:-16}"

# 환경 변수 export
export OPENCLIP_ROOT
export PRETRAINED
export CSV_PATH

# 디렉토리 생성
mkdir -p ${OUTPUT_DIR}
mkdir -p ${LOG_DIR}

# 로그 파일 경로
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/finetune_${TIMESTAMP}.log"

echo "환경 설정:"
echo "  OpenCLIP Root: ${OPENCLIP_ROOT}"
echo "  Pretrained: ${PRETRAINED}"
echo "  CSV Path: ${CSV_PATH}"
echo "  GPU: ${GPU_NUM}"
echo "  Model: ${MODEL_NAME}"
echo "  Experiment Name: ${EXPERIMENT_NAME}"
echo "  Epochs: ${EPOCHS}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Learning Rate: ${LR}"
echo "  Output Directory: ${OUTPUT_DIR}"
echo "  Log File: ${LOG_FILE}"
echo "=========================================="

# 경로 존재 확인
if [ ! -d "${OPENCLIP_ROOT}" ]; then
    echo "❌ 오류: OpenCLIP 디렉토리를 찾을 수 없습니다: ${OPENCLIP_ROOT}"
    exit 1
fi

if [ ! -f "${PRETRAINED}" ]; then
    echo "⚠️  경고: Pretrained 체크포인트를 찾을 수 없습니다: ${PRETRAINED}"
    echo "계속 진행합니다..."
fi

if [ ! -f "${CSV_PATH}" ]; then
    echo "❌ 오류: CSV 파일을 찾을 수 없습니다: ${CSV_PATH}"
    exit 1
fi

# Python 스크립트 실행
echo "Fine-tuning 시작..."
python3 /src/finetune_hnscc.py 2>&1 | tee "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo "=========================================="
    echo "✅ Fine-tuning 완료"
    echo "완료 시간: $(date)"
    echo "로그 파일: ${LOG_FILE}"
    echo "=========================================="
else
    echo "=========================================="
    echo "❌ Fine-tuning 실패 (종료 코드: $EXIT_CODE)"
    echo "실패 시간: $(date)"
    echo "로그 파일: ${LOG_FILE}"
    echo "=========================================="
    exit $EXIT_CODE
fi

