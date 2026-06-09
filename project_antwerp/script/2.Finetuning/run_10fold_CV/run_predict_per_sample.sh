#!/bin/bash
# bash run_predict_per_sample.sh
EMB_ROOT=/project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new
FOLD_DIR=/project_antwerp/hbae/Loki_output/10fold_csv_file/0317_New_HVG_10fold

for FOLD_IDX in 04 05 06 07 08 09 10; do
    FOLD_NAME="fold_${FOLD_IDX}"

    echo "========================================"
    echo "[${FOLD_NAME}] Predicting per sample"
    echo "========================================"

    python predict_per_sample.py \
        --emb_dir ${EMB_ROOT}/${FOLD_NAME} \
        --val_csv ${FOLD_DIR}/${FOLD_NAME}_val.csv \
        --pred_style loki

    if [[ $? -eq 0 ]]; then
        echo "✅ ${FOLD_NAME} done"
    else
        echo "❌ ${FOLD_NAME} failed"
    fi

done

echo "Done!"