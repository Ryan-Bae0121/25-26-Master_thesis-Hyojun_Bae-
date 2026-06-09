#!/bin/bash
# run_loki_predex_10fold.sh
# Loki PredEx 10-fold cross-validation

echo "========================================================================="
echo "Loki PredEx 10-Fold Cross-Validation"
echo "========================================================================="
echo "Start time: $(date)"
echo ""

BASE_DIR="/project_antwerp/hbae/Loki_output"
SCRIPT_DIR="/project_antwerp/hbae/script/0208_start/Validation_hvg_fold"
FOLD_DIR="${BASE_DIR}/folds_10fold_hvg_predex"
OUTPUT_BASE="${BASE_DIR}/loki_predex_10fold"
HVG_FILE="/project_antwerp/hbae/HVG_genelist.txt"

mkdir -p ${OUTPUT_BASE}

# Run all 10 folds
for fold in 01 02 03 04 05 06 07 08 09 10; do
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
echo "All folds completed!"
echo "End time: $(date)"
echo "========================================================================="
echo ""
echo "Aggregating results..."

# Aggregate results
python ${SCRIPT_DIR}/aggregate_loki_predex_results.py \
    --results_dir ${OUTPUT_BASE} \
    --output_file ${OUTPUT_BASE}/10fold_summary.json

echo ""
echo "Results saved to: ${OUTPUT_BASE}/10fold_summary.json"
echo "========================================================================="