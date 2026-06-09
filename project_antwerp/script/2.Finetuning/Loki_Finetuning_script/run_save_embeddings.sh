#!/bin/bash

GT_EXPR=/project_antwerp/hbae/data/0317_HVG_NEW/combined_expression_matrix.npy
GT_OBS=/project_antwerp/hbae/data/0317_HVG_NEW/combined_obs.npy
GENE_LIST=/project_antwerp/hbae/data/0317_HVG_NEW/all_shared_genes.txt
HVG_FILE=/project_antwerp/hbae/data/0317_hvg_2000_list.txt
FOLD_DIR=/project_antwerp/hbae/Loki_output/10fold_csv_file/0317_New_HVG_10fold
FINETUNE_ROOT=/project_antwerp/hbae/Loki_output/0317_10epoch_finetune_10fold_runs_hvg_
OUTPUT_ROOT=/project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new

for FOLD_IDX in 07 08 09 10; do
    FOLD_NAME="fold_${FOLD_IDX}"

    # checkpoint 찾기 (epoch_latest.pt)
    CKPT=$(find ${FINETUNE_ROOT}/${FOLD_NAME} -name "epoch_latest.pt" | head -1)

    if [[ -z "$CKPT" ]]; then
        echo "⚠️  ${FOLD_NAME}: checkpoint not found, skipping"
        continue
    fi

    echo "========================================"
    echo "[${FOLD_NAME}] Saving embeddings"
    echo "  checkpoint: $CKPT"
    echo "========================================"

    python save_embeddings.py \
        --train_csv ${FOLD_DIR}/${FOLD_NAME}_train.csv \
        --val_csv   ${FOLD_DIR}/${FOLD_NAME}_val.csv \
        --hvg_file  ${HVG_FILE} \
        --gt_expr   ${GT_EXPR} \
        --gt_obs    ${GT_OBS} \
        --gene_list ${GENE_LIST} \
        --pretrained ${CKPT} \
        --output_dir ${OUTPUT_ROOT}/${FOLD_NAME} \
        --device cuda:0

    if [[ $? -eq 0 ]]; then
        echo "✅ ${FOLD_NAME} embeddings saved"
    else
        echo "❌ ${FOLD_NAME} failed"
    fi
done

echo "Done!"