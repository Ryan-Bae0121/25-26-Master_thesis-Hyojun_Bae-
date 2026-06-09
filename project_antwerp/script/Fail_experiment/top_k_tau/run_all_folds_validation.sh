#!/bin/bash
# Run ST Validation for all folds (fold 01~10)
# Settings: tau=0.1, top_k=20000, batch_size=10000

TAU=0.1
TOP_K=20000
BATCH_SIZE=10000
VAL_SAMPLE=1000

CKPT_ROOT="/project_antwerp/hbae/Loki_output/finetune_10fold_runs"
FOLD_DIR="/project_antwerp/hbae/Loki_output/folds_10fold"
EXPR_NPY="/project_antwerp/hbae/data/combined_expression_matrix.npy"
GENES_FILE="/project_antwerp/hbae/script/top300_genes.txt"
OUT_ROOT="/project_antwerp/hbae/Loki_output/st_val_final"

echo "========================================================================"
echo "ST Validation - All Folds"
echo "Settings: tau=$TAU, top_k=$TOP_K, batch_size=$BATCH_SIZE"
echo "========================================================================"
echo ""

# Fold 01은 이미 embeddings 있으니까 inference만!
echo "========================================================================"
echo "FOLD 01 - Reusing saved embeddings"
echo "========================================================================"

python3 predex_final_inference.py \
    --fold 1 \
    --ckpt_path "$CKPT_ROOT/fold_01/finetune_fold_01_20260103_003308/checkpoints/epoch_latest.pt" \
    --train_csv "$FOLD_DIR/fold_01_train_fixed.csv" \
    --val_csv "$FOLD_DIR/fold_01_val_fixed.csv" \
    --combined_expr_npy "$EXPR_NPY" \
    --top_genes_file "$GENES_FILE" \
    --saved_embeds_dir "/project_antwerp/hbae/Loki_output/st_val_results/fold_01/train_batches" \
    --tau $TAU \
    --top_k $TOP_K \
    --val_sample_size $VAL_SAMPLE \
    --out_dir "$OUT_ROOT/fold_01" \
    --device cuda

echo ""

# Fold 02~10: encoding + inference
for FOLD_NUM in 02 03 04 05 06 07 08 09 10; do
    echo "========================================================================"
    echo "FOLD $FOLD_NUM - Encoding + Inference"
    echo "========================================================================"

    FOLD_NAME="fold_$FOLD_NUM"

    # Find checkpoint
    CKPT=$(find "$CKPT_ROOT/$FOLD_NAME" -name "epoch_latest.pt" 2>/dev/null | head -1)

    if [ -z "$CKPT" ]; then
        echo "❌ Checkpoint not found for $FOLD_NAME, skipping..."
        continue
    fi

    echo "Checkpoint: $CKPT"

    python3 predex_final_inference.py \
        --fold ${FOLD_NUM#0} \
        --ckpt_path "$CKPT" \
        --train_csv "$FOLD_DIR/${FOLD_NAME}_train_fixed.csv" \
        --val_csv "$FOLD_DIR/${FOLD_NAME}_val_fixed.csv" \
        --combined_expr_npy "$EXPR_NPY" \
        --top_genes_file "$GENES_FILE" \
        --tau $TAU \
        --top_k $TOP_K \
        --batch_size $BATCH_SIZE \
        --val_sample_size $VAL_SAMPLE \
        --out_dir "$OUT_ROOT/$FOLD_NAME" \
        --device cuda

    echo ""
done

# Summary
echo "========================================================================"
echo "FINAL SUMMARY - All Folds"
echo "========================================================================"

python3 - << PYTHON
import json
import numpy as np
from pathlib import Path

out_root = Path("$OUT_ROOT")
results = []

for fold_num in range(1, 11):
    fold_name = f'fold_{fold_num:02d}'
    result_file = out_root / fold_name / 'results.json'

    if result_file.exists():
        with open(result_file) as f:
            r = json.load(f)
        results.append(r)
        print(f"fold_{fold_num:02d}: Spot={r['spot_corr_mean']:.4f} Gene={r['gene_corr_mean']:.4f} Var={r['var_ratio_mean']:.4f}")
    else:
        print(f"fold_{fold_num:02d}: ❌ Not found")

if results:
    spot_corrs = [r['spot_corr_mean'] for r in results]
    gene_corrs = [r['gene_corr_mean'] for r in results]
    var_ratios  = [r['var_ratio_mean'] for r in results]

    print()
    print("="*60)
    print("10-FOLD AVERAGE")
    print("="*60)
    print(f"Spot corr: {np.mean(spot_corrs):.4f} ± {np.std(spot_corrs):.4f}")
    print(f"Gene corr: {np.mean(gene_corrs):.4f} ± {np.std(gene_corrs):.4f}")
    print(f"Var ratio: {np.mean(var_ratios):.4f} ± {np.std(var_ratios):.4f}")
    print("="*60)
PYTHON

