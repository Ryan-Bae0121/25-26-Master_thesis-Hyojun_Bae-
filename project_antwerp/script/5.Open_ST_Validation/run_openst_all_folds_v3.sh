#!/bin/bash
# Open-ST External Validation - FOV Aggregation v2 (올바른 normalize)
# Step 2 (embedding) + Step 3 (predict & eval) for folds 02-10
# Usage: bash run_openst_all_folds.sh

H5_PATH="/project_antwerp/hbae/data/Open_ST/openst_patches_agg_mc10.h5"
HVG_FILE="/project_antwerp/hbae/data/0317_hvg_2000_list.txt"
FINETUNE_BASE="/project_antwerp/hbae/Loki_output/0317_10epoch_finetune_10fold_runs_hvg_"
TRAIN_EMB_BASE="/project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new"
OUT_BASE="/project_antwerp/hbae/Loki_output/openst_validation_agg_v2"
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

    # checkpoint 존재 확인
    if [ ! -f "$CKPT" ]; then
        echo "  WARNING: checkpoint not found: $CKPT"
        echo "  Skipping fold ${FOLD}"
        continue
    fi

    # train embedding 존재 확인
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
        > ${LOG_DIR}/openst_v2_step2_fold${FOLD}.log 2>&1

    if [ $? -ne 0 ]; then
        echo "  ERROR in Step 2 fold ${FOLD}, check: ${LOG_DIR}/openst_v2_step2_fold${FOLD}.log"
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
        > ${LOG_DIR}/openst_v2_step3_fold${FOLD}.log 2>&1

    if [ $? -ne 0 ]; then
        echo "  ERROR in Step 3 fold ${FOLD}, check: ${LOG_DIR}/openst_v2_step3_fold${FOLD}.log"
        continue
    fi
    echo "  Step 3 done."

    # 결과 요약 출력
    echo "  Results:"
    grep "PCC" ${LOG_DIR}/openst_v2_step3_fold${FOLD}.log | grep -v "eval genes"
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
print()
print('=== Open-ST External Validation Summary (FOV Aggregation v2) ===')
print(f'  H5: ${H5_PATH}')
print(f'  Normalize: 전체 gene 기준 normalize -> HVG subset (expm1 sum ~2023)')
print()
print(f"{'Fold':<10} {'Cell-wise PCC':>15} {'Gene-wise PCC':>15} {'Gene-wise median':>18}")
print('-' * 62)

cw_all, gw_all = [], []
for fold in range(1, 11):
    cw_path = f'{base}/fold_{fold:02d}/openst_cellwise_pcc.npy'
    gw_path = f'{base}/fold_{fold:02d}/openst_genewise_pcc.npy'
    if os.path.exists(cw_path) and os.path.exists(gw_path):
        cw = np.load(cw_path)
        gw = np.load(gw_path)
        cw_all.append(cw.mean())
        gw_all.append(gw.mean())
        print(f'fold_{fold:02d}    {cw.mean():>15.4f} {gw.mean():>15.4f} {np.median(gw):>18.4f}')
    else:
        print(f'fold_{fold:02d}    {'N/A':>15} {'N/A':>15} {'N/A':>18}')

if cw_all:
    print('-' * 62)
    print(f'{'Mean':10} {np.mean(cw_all):>15.4f} {np.mean(gw_all):>15.4f}')
    print(f'{'Std':10} {np.std(cw_all):>15.4f} {np.std(gw_all):>15.4f}')
    print()
    print(f'Final: Cell-wise PCC = {np.mean(cw_all):.4f} +/- {np.std(cw_all):.4f}')
    print(f'Final: Gene-wise PCC = {np.mean(gw_all):.4f} +/- {np.std(gw_all):.4f}')
"