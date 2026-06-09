#!/bin/bash
# Open-ST External Validation - FOV Aggregation Pipeline
# Step 2 (embedding) + Step 3 (predict & eval) for folds 02-10
# Usage: bash run_openst_all_folds.sh

H5_PATH="/project_antwerp/hbae/data/Open_ST/openst_patches_agg_mc10.h5"
HVG_FILE="/project_antwerp/hbae/data/0317_hvg_2000_list.txt"
FINETUNE_BASE="/project_antwerp/hbae/Loki_output/0317_10epoch_finetune_10fold_runs_hvg_"
TRAIN_EMB_BASE="/project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new"
OUT_BASE="/project_antwerp/hbae/Loki_output/openst_validation_agg"
SCRIPT_DIR="/project_antwerp/hbae/script/0208_start/Open_ST_Validation"
LOG_DIR="/project_antwerp/hbae/logs"

mkdir -p $LOG_DIR

declare -A CKPT_MAP
CKPT_MAP["02"]="finetune_hvg_fold_02_20260320_224414"
CKPT_MAP["03"]="finetune_hvg_fold_03_20260321_000327"
CKPT_MAP["04"]="finetune_hvg_fold_04_20260323_194045"
CKPT_MAP["05"]="finetune_hvg_fold_05_20260323_205941"
CKPT_MAP["06"]="finetune_hvg_fold_06_20260323_221515"
CKPT_MAP["07"]="finetune_hvg_fold_07_20260324_095306"
CKPT_MAP["08"]="finetune_hvg_fold_08_20260324_144021"
CKPT_MAP["09"]="finetune_hvg_fold_09_20260324_163351"
CKPT_MAP["10"]="finetune_hvg_fold_10_20260324_183036"

for FOLD in 02 03 04 05 06 07 08 09 10; do
    echo "=============================="
    echo "Processing fold ${FOLD} ..."
    echo "=============================="

    CKPT="${FINETUNE_BASE}/fold_${FOLD}/${CKPT_MAP[$FOLD]}/checkpoints/epoch_latest.pt"
    OUT_EMB_DIR="${OUT_BASE}/fold_${FOLD}"
    TRAIN_EMB_DIR="${TRAIN_EMB_BASE}/fold_${FOLD}"

    if [ ! -f "$CKPT" ]; then
        echo "  WARNING: checkpoint not found: $CKPT"
        echo "  Skipping fold ${FOLD}"
        continue
    fi

    if [ ! -f "${TRAIN_EMB_DIR}/train_img_embs.npy" ]; then
        echo "  WARNING: train embeddings not found: ${TRAIN_EMB_DIR}"
        echo "  Skipping fold ${FOLD}"
        continue
    fi

    # Step 2: Embedding
    echo "[Step 2] Embedding fold ${FOLD} ..."
    CUDA_VISIBLE_DEVICES=0 python ${SCRIPT_DIR}/openst_step2_embed.py \
        --h5_path    ${H5_PATH} \
        --pretrained ${CKPT} \
        --output_dir ${OUT_EMB_DIR} \
        --device     cuda:0 \
        > ${LOG_DIR}/openst_agg_step2_fold${FOLD}.log 2>&1

    if [ $? -ne 0 ]; then
        echo "  ERROR in Step 2 fold ${FOLD}"
        continue
    fi
    echo "  Step 2 done."

    # Step 3: Predict & Eval
    echo "[Step 3] Predict fold ${FOLD} ..."
    python ${SCRIPT_DIR}/openst_step3_predict_eval.py \
        --openst_emb_dir ${OUT_EMB_DIR} \
        --train_emb_dir  ${TRAIN_EMB_DIR} \
        --h5_path        ${H5_PATH} \
        --hvg_file       ${HVG_FILE} \
        --pred_style loki \
        --top_k 50 \
        > ${LOG_DIR}/openst_agg_step3_fold${FOLD}.log 2>&1

    if [ $? -ne 0 ]; then
        echo "  ERROR in Step 3 fold ${FOLD}"
        continue
    fi
    echo "  Step 3 done."

    # 결과 요약
    grep "PCC" ${LOG_DIR}/openst_agg_step3_fold${FOLD}.log
    echo ""
done

echo "=============================="
echo "All folds done!"
echo "=============================="

# 전체 결과 요약
python3 -c "
import numpy as np
import os

base = '${OUT_BASE}'
print('\n=== Final Summary across all folds ===')
print(f'{'Fold':<8} {'Cell-wise PCC':>15} {'Gene-wise PCC':>15}')
print('-' * 40)

cw_all, gw_all = [], []
for fold in range(1, 11):
    cw = f'{base}/fold_{fold:02d}/openst_cellwise_pcc.npy'
    gw = f'{base}/fold_{fold:02d}/openst_genewise_pcc.npy'
    if os.path.exists(cw) and os.path.exists(gw):
        cw_arr = np.load(cw)
        gw_arr = np.load(gw)
        cw_all.append(cw_arr.mean())
        gw_all.append(gw_arr.mean())
        print(f'fold_{fold:02d}  {cw_arr.mean():>15.4f} {gw_arr.mean():>15.4f}')

if cw_all:
    print('-' * 40)
    print(f'{'Mean':<8} {np.mean(cw_all):>15.4f} {np.mean(gw_all):>15.4f}')
    print(f'{'Std':<8} {np.std(cw_all):>15.4f} {np.std(gw_all):>15.4f}')
"