#!/bin/bash
# Loki HNSCC finetuning 실행 스크립트 (GPULab / Docker용)

set -e  # 에러 나면 바로 중단

echo "=========================================="
echo "Loki HNSCC finetuning 시작"
echo "=========================================="

#########################
# 경로 설정 (필요하면 수정)
#########################

# open_clip2 레포 위치 (스토리지에 있는 거 그대로 사용)
OPENCLIP_ROOT="${OPENCLIP_ROOT:-/project_antwerp/hbae/open_clip2}"

# finetune_hnscc.py 위치
WSI_RNA_ROOT="${WSI_RNA_ROOT:-/project_antwerp/hbae/WSI-RNA-LOKI2}"
SRC_DIR="${SRC_DIR:-${WSI_RNA_ROOT}/src}"

# 데이터 / 출력 / 사전학습 체크포인트
DATA_ROOT="${DATA_ROOT:-/project_antwerp/hbae/Loki_data}"
OUTPUT_DIR="${OUTPUT_DIR:-/project_antwerp/hbae/Loki_output}"
CKPT_PATH="${CKPT_PATH:-/project_antwerp/assets/loki_ckpts/checkpoint.pt}"

# 필요시 조절 가능한 하이퍼파라미터 (env로 override 가능)
export EPOCHS="${EPOCHS:-5}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export LR="${LR:-5e-6}"
export WD="${WD:-0.1}"
export WARMUP="${WARMUP:-10}"
export WORKERS="${WORKERS:-4}"
export MODEL_NAME="${MODEL_NAME:-coca_ViT-L-14}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-finetune_HEG_hnscc_$(date +%s)}"

# GPU 설정 (GPULab이 알아서 해주지만, 필요시 override)
GPU_NUM="${GPU_NUM:-0}"
export CUDA_VISIBLE_DEVICES=$GPU_NUM

# WandB API 키
export WANDB_API_KEY="${WANDB_API_KEY:-a8d69bdb4fd712911b58b798d2045e86cd34a4d0}"

#########################
# PYTHONPATH / 작업 디렉토리
#########################

export PYTHONPATH="${OPENCLIP_ROOT}/src:${PYTHONPATH}"

echo "OPENCLIP_ROOT : ${OPENCLIP_ROOT}"
echo "WSI_RNA_ROOT  : ${WSI_RNA_ROOT}"
echo "SRC_DIR       : ${SRC_DIR}"
echo "DATA_ROOT     : ${DATA_ROOT}"
echo "OUTPUT_DIR    : ${OUTPUT_DIR}"
echo "CKPT_PATH     : ${CKPT_PATH}"
echo "GPU           : ${GPU_NUM}"
echo "BATCH_SIZE    : ${BATCH_SIZE}"
echo "EPOCHS        : ${EPOCHS}"
echo "EXPERIMENT    : ${EXPERIMENT_NAME}"
echo "=========================================="

# open_clip 루트에서 실행 (finetune_hnscc.py 안에서 -m open_clip_train.main 호출)
cd "${OPENCLIP_ROOT}"

python -u "${SRC_DIR}/finetune_hnscc.py" \
  --data_root "${DATA_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --checkpoint_path "${CKPT_PATH}"

RETVAL=$?

echo "=========================================="
echo "Loki HNSCC finetuning 종료 (exit code: ${RETVAL})"
echo "=========================================="

exit ${RETVAL}
