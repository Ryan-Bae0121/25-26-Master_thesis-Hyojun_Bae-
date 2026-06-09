#!/bin/bash
# ============================================================
# TCGA Bulk Prediction - fold_01 ~ fold_10 전체 실행
# step1: 이전 스크립트 그대로 (fold_01~10 순차 embedding 추출)
# step2: 새 스크립트 (top 300 방법 A, B 포함 PCC 계산)
# ============================================================

SCRIPT_DIR="/project_antwerp/hbae/script/0208_start/Bulk_analysis"
LOG_DIR="/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/logs"
mkdir -p $LOG_DIR

echo "============================================"
echo "TCGA Bulk Prediction Pipeline 시작"
echo "시작 시간: $(date)"
echo "============================================"

# ── Step 1: fold_01~10 embedding 추출 (이전 스크립트 그대로) ──
echo ""
echo "[Step 1] Embedding 추출 시작..."
echo "Log: $LOG_DIR/step1.log"
echo ""

CUDA_VISIBLE_DEVICES=0 python $SCRIPT_DIR/step1_extract_tcga_embeddings.py \
    2>&1 | tee $LOG_DIR/step1.log

EXIT_CODE=${PIPESTATUS[0]}
if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "[Step 1] ERROR! Exit code: $EXIT_CODE"
    echo "Log 확인: $LOG_DIR/step1.log"
    exit 1
fi

echo ""
echo "============================================"
echo "[Step 1] 완료: $(date)"
echo "============================================"

# ── Step 2: PredEx + top300 PCC 계산 (새 스크립트) ──
echo ""
echo "[Step 2] PredEx + Ensemble PCC 계산 시작..."
echo "Log: $LOG_DIR/step2.log"
echo ""

CUDA_VISIBLE_DEVICES=0 python $SCRIPT_DIR/step2_predict_from_embeddings.py \
    2>&1 | tee $LOG_DIR/step2.log

EXIT_CODE=${PIPESTATUS[0]}
if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "[Step 2] ERROR! Exit code: $EXIT_CODE"
    echo "Log 확인: $LOG_DIR/step2.log"
    exit 1
fi

echo ""
echo "============================================"
echo "전체 파이프라인 완료: $(date)"
echo "============================================"