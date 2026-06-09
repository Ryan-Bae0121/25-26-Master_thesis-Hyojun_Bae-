#!/bin/bash

# Fine-tuning 실행 스크립트
# Usage: bash run_finetuning.sh

set -e  # 에러 발생 시 스크립트 중단

# 기본 경로 설정 (필요에 따라 수정)
DATA_DIR="${DATA_DIR:-/project/data}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/project/checkpoints}"
OUTPUT_DIR="${OUTPUT_DIR:-/project/output}"
SRC_DIR="${SRC_DIR:-/src}"

# GPU 설정
GPU_NUM="${GPU_NUM:-0}"
export CUDA_VISIBLE_DEVICES=$GPU_NUM

# Fine-tuning 하이퍼파라미터
BATCH_SIZE="${BATCH_SIZE:-32}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
NUM_EPOCHS="${NUM_EPOCHS:-50}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# 모델 설정
MODEL_NAME="${MODEL_NAME:-ViT-B/32}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-finetuning_exp}"

# 데이터 경로
TRAIN_DATA="${TRAIN_DATA:-${DATA_DIR}/train}"
VAL_DATA="${VAL_DATA:-${DATA_DIR}/val}"

# 로그 디렉토리
LOG_DIR="${OUTPUT_DIR}/logs/${EXPERIMENT_NAME}"
mkdir -p ${LOG_DIR}

# 출력 디렉토리 생성
mkdir -p ${OUTPUT_DIR}
mkdir -p ${CHECKPOINT_DIR}

echo "=========================================="
echo "Fine-tuning 시작"
echo "=========================================="
echo "GPU: ${GPU_NUM}"
echo "Data Directory: ${DATA_DIR}"
echo "Checkpoint Directory: ${CHECKPOINT_DIR}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Model: ${MODEL_NAME}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Learning Rate: ${LEARNING_RATE}"
echo "Epochs: ${NUM_EPOCHS}"
echo "Experiment Name: ${EXPERIMENT_NAME}"
echo "=========================================="

# Python 스크립트 실행
# src 디렉토리에 finetune.py 또는 train.py가 있다고 가정
# 실제 파일명에 맞게 수정 필요
python3 ${SRC_DIR}/finetune.py \
    --data_dir ${DATA_DIR} \
    --train_data ${TRAIN_DATA} \
    --val_data ${VAL_DATA} \
    --checkpoint_dir ${CHECKPOINT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --pretrained_checkpoint ${PRETRAINED_CHECKPOINT} \
    --model_name ${MODEL_NAME} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --num_epochs ${NUM_EPOCHS} \
    --num_workers ${NUM_WORKERS} \
    --experiment_name ${EXPERIMENT_NAME} \
    --gpu_num ${GPU_NUM} \
    2>&1 | tee ${LOG_DIR}/training.log

echo "=========================================="
echo "Fine-tuning 완료"
echo "로그 파일: ${LOG_DIR}/training.log"
echo "=========================================="

