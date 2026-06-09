#!/bin/bash
# Loki HNSCC 10-fold CV finetuning 실행 스크립트 (GPULab / Docker용)
# This script runs all three steps in sequence:
# 1. Create fold splits
# 2. Run fine-tuning for each fold
# 3. Collect checkpoint paths

set -e  # 에러 나면 바로 중단

echo "=========================================="
echo "Loki HNSCC 10-fold CV finetuning 시작"
echo "=========================================="

#########################
# 경로 설정 (필요하면 수정)
#########################

# open_clip2 레포 위치 (스토리지에 있는 거 그대로 사용)
OPENCLIP_ROOT="${OPENCLIP_ROOT:-/project_antwerp/hbae/open_clip2}"

# finetune scripts 위치
WSI_RNA_ROOT="${WSI_RNA_ROOT:-/project_antwerp/hbae/WSI-RNA-LOKI2}"
SRC_DIR="${SRC_DIR:-${WSI_RNA_ROOT}/src}"

# 데이터 / 출력 / 사전학습 체크포인트
DATA_ROOT="${DATA_ROOT:-/project_antwerp/hbae/Loki_data}"
META_CSV="${META_CSV:-/project_antwerp/hbae/data/HEG_finetune_meta_gpulab.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-/project_antwerp/hbae/Loki_output}"
CKPT_PATH="${CKPT_PATH:-/project_antwerp/assets/loki_ckpts/checkpoint.pt}"

# 10-fold 관련 설정
FOLD_DIR="${FOLD_DIR:-${OUTPUT_DIR}/folds_10fold}"
OUT_ROOT="${OUT_ROOT:-${OUTPUT_DIR}/finetune_10fold_runs}"
N_FOLDS="${N_FOLDS:-10}"
SEED="${SEED:-42}"

# 필요시 조절 가능한 하이퍼파라미터 (env로 override 가능)
export EPOCHS="${EPOCHS:-10}"  # 논문: Loki PredEx는 10 epochs 사용
export BATCH_SIZE="${BATCH_SIZE:-64}"
export LR="${LR:-5e-6}"
export WD="${WD:-0.1}"
export WARMUP="${WARMUP:-10}"
export WORKERS="${WORKERS:-4}"
export MODEL_NAME="${MODEL_NAME:-coca_ViT-L-14}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-finetune_HEG_hnscc_10fold_$(date +%s)}"

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
echo "META_CSV      : ${META_CSV}"
echo "OUTPUT_DIR    : ${OUTPUT_DIR}"
echo "FOLD_DIR      : ${FOLD_DIR}"
echo "OUT_ROOT      : ${OUT_ROOT}"
echo "CKPT_PATH     : ${CKPT_PATH}"
echo "GPU           : ${GPU_NUM}"
echo "BATCH_SIZE    : ${BATCH_SIZE}"
echo "EPOCHS        : ${EPOCHS}"
echo "N_FOLDS       : ${N_FOLDS}"
echo "SEED          : ${SEED}"
echo "EXPERIMENT    : ${EXPERIMENT_NAME}"
echo "=========================================="

#########################
# Step 1: Create fold splits
#########################
echo ""
echo "=========================================="
echo "Step 1: Creating fold splits"
echo "=========================================="

cd "${SRC_DIR}"

python -u "${SRC_DIR}/make_folds_and_csvs.py" \
    --meta_csv "${META_CSV}" \
    --out_dir "${FOLD_DIR}" \
    --n_folds "${N_FOLDS}" \
    --seed "${SEED}" \
    --img_col img_path \
    --label_col label \
    --slide_col sample_name \
    --output_img_col filepath \
    --output_label_col title \
    --overwrite

echo ""
echo "✅ Fold splits created!"
echo ""

#########################
# Step 2: Run fine-tuning for each fold
#########################
echo "=========================================="
echo "Step 2: Running fine-tuning for each fold"
echo "=========================================="

python -u "${SRC_DIR}/run_finetune_10fold.py" \
    --fold_dir "${FOLD_DIR}" \
    --openclip_root "${OPENCLIP_ROOT}" \
    --pretrained_ckpt "${CKPT_PATH}" \
    --model "${MODEL_NAME}" \
    --out_root "${OUT_ROOT}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --continue_on_error

echo ""
echo "✅ Fine-tuning completed!"
echo ""

#########################
# Step 3: Collect checkpoint paths
#########################
echo "=========================================="
echo "Step 3: Collecting checkpoint paths"
echo "=========================================="

python -u "${SRC_DIR}/collect_finetune_ckpts.py" \
    --out_root "${OUT_ROOT}" \
    --pattern "*most_recent*.pt"

echo ""
echo "✅ Checkpoint collection completed!"
echo ""

#########################
# Step 4: Analyze fold performance
#########################
echo "=========================================="
echo "Step 4: Analyzing fold performance"
echo "=========================================="

# Metric to analyze (default: cmc_r10, can be overridden)
METRIC="${METRIC:-cmc_r10}"

python -u "${SRC_DIR}/analyze_fold_performance.py" \
    --out_root "${OUT_ROOT}" \
    --metric "${METRIC}"

echo ""
echo "✅ Performance analysis completed!"
echo ""

#########################
# Summary
#########################
echo "=========================================="
echo "Pipeline Summary"
echo "=========================================="
echo "Fold directory: ${FOLD_DIR}"
echo "Output root: ${OUT_ROOT}"
echo ""
echo "Results:"
echo "  1. Checkpoint index: ${OUT_ROOT}/ckpt_index.csv"
echo "  2. Performance analysis: ${OUT_ROOT}/fold_performance.csv"
echo ""
echo "Next steps:"
echo "  1. Check ${OUT_ROOT}/fold_performance.csv for best fold"
echo "  2. Use best fold checkpoint for evaluation or inference"
echo "=========================================="

RETVAL=$?

echo "=========================================="
echo "Loki HNSCC 10-fold CV finetuning 종료 (exit code: ${RETVAL})"
echo "=========================================="

exit ${RETVAL}

