#!/bin/bash
# Open-ST External Validation - Step 2 & 3 전체 fold 실행
# Usage: bash run_openst_all_folds.sh

H5_PATH="/project_antwerp/hbae/data/Open_ST/openst_patches_level3.h5"
HVG_FILE="/project_antwerp/hbae/data/0317_hvg_2000_list.txt"
FINETUNE_BASE="/project_antwerp/hbae/Loki_output/0317_10epoch_finetune_10fold_runs_hvg_"
TRAIN_EMB_BASE="/project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new"
OUT_BASE="/project_antwerp/hbae/Loki_output/openst_validation_level3"
SCRIPT_DIR="/project_antwerp/hbae/script/0208_start/Open_ST_Validation"
LOG_DIR="/project_antwerp/hbae/logs"

mkdir -p $LOG_DIR

# fold별 checkpoint 경로 매핑
declare -A CKPT_MAP
CKPT_MAP["01"]="finetune_hvg_fold_01_20260320_212457"
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
        > ${LOG_DIR}/openst_step2_fold${FOLD}.log 2>&1

    if [ $? -ne 0 ]; then
        echo "  ERROR in Step 2 fold ${FOLD}, check log"
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
        > ${LOG_DIR}/openst_step3_fold${FOLD}.log 2>&1

    if [ $? -ne 0 ]; then
        echo "  ERROR in Step 3 fold ${FOLD}, check log"
        continue
    fi
    echo "  Step 3 done."
    echo ""

    # 결과 요약 출력
    tail -10 ${LOG_DIR}/openst_step3_fold${FOLD}.log | grep "PCC"
done

echo "=============================="
echo "All folds done!"
echo "=============================="

# 전체 결과 요약
echo ""
echo "=== Summary across folds ==="
python3 -c "
import pandas as pd
import numpy as np
import os

base = '${OUT_BASE}'
results = []
for fold in range(1, 11):
    csv = f'{base}/fold_{fold:02d}/openst_genewise_pcc.csv'
    cw  = f'{base}/fold_{fold:02d}/openst_cellwise_pcc.npy'
    gw  = f'{base}/fold_{fold:02d}/openst_genewise_pcc.npy'
    if os.path.exists(cw) and os.path.exists(gw):
        cellwise = np.load(cw)
        genewise = np.load(gw)
        results.append({
            'fold': fold,
            'cellwise_pcc_mean': cellwise.mean(),
            'genewise_pcc_mean': genewise.mean(),
            'genewise_pcc_median': np.median(genewise),
        })

if results:
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print()
    print(f'Mean Cell-wise PCC: {df.cellwise_pcc_mean.mean():.4f} ± {df.cellwise_pcc_mean.std():.4f}')
    print(f'Mean Gene-wise PCC: {df.genewise_pcc_mean.mean():.4f} ± {df.genewise_pcc_mean.std():.4f}')
"