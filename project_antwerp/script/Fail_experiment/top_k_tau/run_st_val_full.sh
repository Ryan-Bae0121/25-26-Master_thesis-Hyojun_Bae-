#!/bin/bash

FOLD_NUM=1
FOLD_NAME="fold_01"

echo "================================================================================"
echo "ST Validation - FULL TRAIN SET"
echo "================================================================================"
echo "Fold: $FOLD_NUM"
echo "Expected time: 2-3 hours"
echo "================================================================================"
echo ""

python3 predex_st_validation_full.py \
    --fold $FOLD_NUM \
    --ckpt_path "/project_antwerp/hbae/Loki_output/finetune_10fold_runs/${FOLD_NAME}/finetune_${FOLD_NAME}_20260103_003308/checkpoints/epoch_latest.pt" \
    --train_csv "/project_antwerp/hbae/Loki_output/folds_10fold/${FOLD_NAME}_train_fixed.csv" \
    --val_csv "/project_antwerp/hbae/Loki_output/folds_10fold/${FOLD_NAME}_val_fixed.csv" \
    --combined_expr_npy "/project_antwerp/hbae/data/combined_expression_matrix.npy" \
    --top_genes_file "/project_antwerp/hbae/script/top300_genes.txt" \
    --tau 0.01 \
    --top_k 100 \
    --out_dir "/project_antwerp/hbae/Loki_output/st_val_results/${FOLD_NAME}" \
    --device cuda \
    --save_embeddings

echo ""
echo "================================================================================"
echo "Complete!"
echo "================================================================================"

