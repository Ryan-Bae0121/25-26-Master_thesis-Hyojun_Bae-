#!/bin/bash
# run_loki_predex_folds_2_10_with_preds.sh
# Fold 2-10에 대해 예측값 저장하며 실행

echo "========================================================================="
echo "Loki PredEx Fold 2-10 with Predictions Saving"
echo "========================================================================="
echo "Start time: $(date)"
echo ""

BASE_DIR="/project_antwerp/hbae/Loki_output"
SCRIPT_DIR="/project_antwerp/hbae/script/0208_start/Validation_hvg_fold"
FOLD_DIR="${BASE_DIR}/folds_10fold_hvg_predex"
OUTPUT_BASE="${BASE_DIR}/loki_predex_with_preds"
HVG_FILE="/project_antwerp/hbae/HVG_genelist.txt"

mkdir -p ${OUTPUT_BASE}

# Fold 2-10만 실행 (Fold 01은 이미 완료)
for fold in 02 03 04 05 06 07 08 09 10; do
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Fold ${fold}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Start: $(date)"
    
    python ${SCRIPT_DIR}/loki_predex_exact.py \
        --train_csv ${FOLD_DIR}/fold_${fold}_train.csv \
        --val_csv ${FOLD_DIR}/fold_${fold}_val.csv \
        --hvg_file ${HVG_FILE} \
        --output_dir ${OUTPUT_BASE}/fold_${fold} \
        --temperature 0.07 \
        --save_predictions \
        --device cuda:0
    
    if [ $? -eq 0 ]; then
        echo "✓ Fold ${fold} completed successfully"
    else
        echo "✗ Fold ${fold} failed"
        exit 1
    fi
    
    echo "End: $(date)"
done

echo ""
echo "========================================================================="
echo "All folds (2-10) completed!"
echo "End time: $(date)"
echo "========================================================================="

# Per-gene 분석도 자동 실행
echo ""
echo "Running per-gene analysis for all folds..."
echo ""

for fold in 01 02 03 04 05 06 07 08 09 10; do
    echo "Analyzing fold ${fold}..."
    
    python ${SCRIPT_DIR}/analyze_gene_performance_real.py \
        --predictions_file ${OUTPUT_BASE}/fold_${fold}/predictions.npy \
        --ground_truth_file ${OUTPUT_BASE}/fold_${fold}/ground_truth.npy \
        --hvg_file ${HVG_FILE} \
        --output_dir ${OUTPUT_BASE}/gene_analysis_fold_${fold}
    
    if [ $? -eq 0 ]; then
        echo "  ✓ Analysis completed"
    else
        echo "  ✗ Analysis failed"
    fi
done

echo ""
echo "========================================================================="
echo "All analyses completed!"
echo "========================================================================="