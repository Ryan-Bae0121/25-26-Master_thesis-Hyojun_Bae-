#!/bin/bash
# HVG 기반 10-fold CV fine-tuning (Fixed version)
set -e
# pip install braceexpand
# pip install webdataset  bash run_finetune_10fold_hvg.sh
# ============================================================
# Configuration
# ============================================================
FOLD_DIR=/project_antwerp/hbae/Loki_output/10fold_csv_file/65um_fold_03
OPENCLIP_ROOT=/project_antwerp/hbae/open_clip2
PRETRAINED_CKPT=/project_antwerp/assets/loki_ckpts/checkpoint.pt
OUT_ROOT=/project_antwerp/hbae/Loki_output/0502_65um_finetune
N_FOLDS=10
EPOCHS=10 #원래 10으로 체크용: 2
BATCH_SIZE=64
LR=5e-6
START_FOLD=
END_FOLD=10

echo "========================================================================"
echo "HVG 10-fold CV Fine-tuning (Fixed)"
echo "Fold dir: $FOLD_DIR"
echo "Output: $OUT_ROOT"
echo "Batch size: $BATCH_SIZE, Epochs: $EPOCHS"
echo "========================================================================"

# ============================================================
# Step 1: Fine-tuning each fold
# ============================================================
echo ""
echo "Step 1: Fine-tuning each fold..."

export PYTHONPATH=${OPENCLIP_ROOT}/src:$PYTHONPATH

for FOLD_IDX in $(seq $START_FOLD $END_FOLD); do
    FOLD_NAME=$(printf "fold_%02d" $FOLD_IDX)
    TRAIN_CSV=${FOLD_DIR}/${FOLD_NAME}_train.csv
    VAL_CSV=${FOLD_DIR}/${FOLD_NAME}_val.csv
    
    # CSV 존재 확인
    if [[ ! -f "$TRAIN_CSV" ]]; then
        echo "⚠️  ${FOLD_NAME}: train CSV not found, skipping"
        continue
    fi
    if [[ ! -f "$VAL_CSV" ]]; then
        echo "⚠️  ${FOLD_NAME}: val CSV not found, skipping"
        continue
    fi
    
    EXPERIMENT_NAME="finetune_hvg_${FOLD_NAME}_$(date +%Y%m%d_%H%M%S)"
    FOLD_OUT_DIR=${OUT_ROOT}/${FOLD_NAME}
    mkdir -p ${FOLD_OUT_DIR}
    
    echo ""
    echo "========================================================================"
    echo "[${FOLD_NAME}] Starting fine-tuning"
    echo "  Train: $TRAIN_CSV"
    echo "  Val:   $VAL_CSV"
    echo "  Output: $FOLD_OUT_DIR"
    echo "========================================================================"
    
    cd ${OPENCLIP_ROOT}
    
    python -u -m open_clip_train.main \
        --name ${EXPERIMENT_NAME} \
        --logs ${FOLD_OUT_DIR} \
        --train-data ${TRAIN_CSV} \
        --val-data ${VAL_CSV} \
        --csv-img-key img_path \
        --csv-caption-key label \
        --csv-separator , \
        --model coca_ViT-L-14 \
        --pretrained ${PRETRAINED_CKPT} \
        --device cuda:0 \
        --epochs ${EPOCHS} \
        --batch-size ${BATCH_SIZE} \
        --lr ${LR} \
        --wd 0.1 \
        --warmup 10 \
        --workers 8 \
        --precision fp32 \
        --lock-text-freeze-layer-norm \
        --lock-image-freeze-bn-stats \
        --coca-caption-loss-weight 0 \
        --coca-contrastive-loss-weight 1 \
        --save-frequency 1 \
        --val-frequency 1 \
        --report-to none \
        --debug \
        --save-most-recent \
        2>&1 | tee ${FOLD_OUT_DIR}/finetune.log
    
    EXIT_CODE=${PIPESTATUS[0]}
    
    if [[ $EXIT_CODE -eq 0 ]]; then
        echo "✅ ${FOLD_NAME} completed successfully"
    else
        echo "❌ ${FOLD_NAME} failed with exit code $EXIT_CODE"
    fi
    
    echo ""
done

echo ""
echo "========================================================================"
echo "Fine-tuning completed!"
echo "========================================================================"

# ============================================================
# Step 2: Collect checkpoints
# ============================================================
echo ""
echo "Step 2: Collecting checkpoints..."

python3 - <<'PY'
import json
from pathlib import Path

out_root = Path("/project_antwerp/hbae/Loki_output/0228_finetune_10fold_runs_hvg_v2")
ckpts = []

for fold_dir in sorted(out_root.glob("fold_*")):
    if not fold_dir.is_dir():
        continue
    
    fold_name = fold_dir.name
    
    # epoch_latest.pt 찾기
    ckpt_files = list(fold_dir.rglob("epoch_latest.pt"))
    if not ckpt_files:
        ckpt_files = list(fold_dir.rglob("*.pt"))
    
    if ckpt_files:
        # 가장 최근 파일
        ckpt_path = max(ckpt_files, key=lambda p: p.stat().st_mtime)
        ckpts.append({
            "fold": fold_name,
            "checkpoint": str(ckpt_path),
        })
        print(f"✅ {fold_name}: {ckpt_path}")
    else:
        print(f"⚠️  {fold_name}: No checkpoint found")

# Save checkpoint list
ckpt_list_path = out_root / "checkpoint_list.json"
with open(ckpt_list_path, 'w') as f:
    json.dump(ckpts, f, indent=2)

print(f"\n✅ Checkpoint list saved: {ckpt_list_path}")
print(f"   Total: {len(ckpts)} / 10 folds")
PY

echo ""
echo "========================================================================"
echo "Done!"
echo "  Checkpoints: ${OUT_ROOT}/checkpoint_list.json"
echo "========================================================================"