#!/bin/bash
# Open-ST Scanpy HVG300 Evaluation - All Folds (01-10)
# Usage: bash run_openst_hvg_all_folds.sh

SCRIPT_DIR="/project_antwerp/hbae/script/0208_start/Open_ST_Validation"
H5_PATH="/project_antwerp/hbae/data/Open_ST/openst_patches_agg_mc10.h5"
HVG_FILE="/project_antwerp/hbae/data/0317_hvg_2000_list.txt"
EMB_BASE="/project_antwerp/hbae/Loki_output/openst_validation_agg_v2"
TRAIN_EMB_BASE="/project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new"
OUT_BASE="/project_antwerp/hbae/Loki_output/openst_validation_hvg300"
LOG_DIR="/project_antwerp/hbae/logs"

mkdir -p $LOG_DIR $OUT_BASE

for FOLD in 01 02 03 04 05 06 07 08 09 10; do
    echo "=============================="
    echo "HVG300 eval fold ${FOLD} ..."
    echo "=============================="

    OPENST_EMB="${EMB_BASE}/fold_${FOLD}"
    TRAIN_EMB="${TRAIN_EMB_BASE}/fold_${FOLD}"

    if [ ! -f "${OPENST_EMB}/openst_img_embs.npy" ]; then
        echo "  WARNING: embedding not found: ${OPENST_EMB}"
        echo "  Skipping fold ${FOLD}"
        continue
    fi
    if [ ! -f "${TRAIN_EMB}/train_img_embs.npy" ]; then
        echo "  WARNING: train embedding not found: ${TRAIN_EMB}"
        echo "  Skipping fold ${FOLD}"
        continue
    fi

    python ${SCRIPT_DIR}/openst_step3_hvgpredict_eval.py \
        --openst_emb_dir ${OPENST_EMB} \
        --train_emb_dir  ${TRAIN_EMB} \
        --h5_path        ${H5_PATH} \
        --hvg_file       ${HVG_FILE} \
        --pred_style loki \
        --gene_select scanpy_hvg \
        > ${LOG_DIR}/openst_hvg_fold${FOLD}.log 2>&1

    if [ $? -ne 0 ]; then
        echo "  ERROR fold ${FOLD}, check: ${LOG_DIR}/openst_hvg_fold${FOLD}.log"
        continue
    fi

    echo "  Done."
    grep -E "Cell-wise|Gene-wise PCC" ${LOG_DIR}/openst_hvg_fold${FOLD}.log
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
print('=== Open-ST Scanpy HVG300 Evaluation Summary ===')
print(f'  방식: Scanpy seurat flavor HVG top-300')
print()
print(f'{'Fold':<10} {'Cell-wise PCC':>15} {'Gene-wise PCC':>15} {'Gene-wise median':>18}')
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