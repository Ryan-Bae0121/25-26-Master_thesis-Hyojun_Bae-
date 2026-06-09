#!/usr/bin/env bash
# Use existing checkpoint.pt for zero-shot pipeline

set -euo pipefail

echo "=================================================================="
echo "Zero-Shot Pipeline with Existing Checkpoint"
echo "Slide: TCGA-QK-A6IH-01Z-00-DX1"
echo "=================================================================="

# Configuration
ROOT=/home/students/hbae/Loki/zero_shot_pipeline
CKPT=/home/students/hbae/Loki/data/basic_usage/checkpoint.pt
TILES_DIR=/home/students/hbae/Loki/patches_hdf5_with_annot/TCGA-QK-A6IH-01Z-00-DX1
BANK_H5AD=/home/students/hbae/Loki/hest_hnscc_subset/hnscc_only_2patients/expression_log1p.h5ad
EVAL_SCRIPT=/home/students/hbae/Loki/slide_level_calibrated_metrics.py
TCGA_CSV=/shares/bioit-students/ir_students_2020/TCGA-HNSC/ref_file.csv
SLIDE_ID=TCGA-QK-A6IH-01Z-00-DX1
OUT_ROOT=${ROOT}/zero_shot_QK_A6IH_EXISTING

cd "${ROOT}"

echo ""
echo "Configuration:"
echo "  Checkpoint: $CKPT"
echo "  Tiles: $TILES_DIR"
echo "  Bank: $BANK_H5AD"
echo "  Output: $OUT_ROOT"
echo ""

# Verify checkpoint exists
if [ ! -f "$CKPT" ]; then
    echo "❌ Checkpoint not found: $CKPT"
    exit 1
fi

echo "✅ Checkpoint exists ($(ls -lh $CKPT | awk '{print $5}'))"
echo ""

# Create checkpoint directory and copy
mkdir -p "${OUT_ROOT}/checkpoint"
cp "$CKPT" "${OUT_ROOT}/checkpoint/checkpoint.pt"

echo "=================================================================="
echo "Step 1/2: Building Bank and Running Zero-Shot Prediction"
echo "=================================================================="

# Build bank
echo "[INFO] Building text bank..."
python build_text_bank.py \
  --bank_h5ad "${BANK_H5AD}" \
  --hf_ckpt "$CKPT" \
  --out_dir "${OUT_ROOT}/bank" \
  --topn 50 \
  --device cpu \
  --batch_size 64

echo ""
echo "[INFO] Running zero-shot prediction..."
python zero_shot_loki.py \
  --tiles_dir "${TILES_DIR}" \
  --bank_text_emb "npy:${OUT_ROOT}/bank/bank_text_emb.npy" \
  --bank_expr "npy:${OUT_ROOT}/bank/bank_expr.npy" \
  --bank_genes "${OUT_ROOT}/bank/bank_genes.txt" \
  --out_csv "${OUT_ROOT}/pred/pred_tile_gene.csv" \
  --hf_ckpt "$CKPT" \
  --device cpu \
  --batch_size 128 \
  --normalize bank_log1p \
  --temp 1.0 \
  --topk 64

echo ""
echo "=================================================================="
echo "Step 2/2: Slide-Level Evaluation"
echo "=================================================================="

python "${EVAL_SCRIPT}" \
  --pred_csv "${OUT_ROOT}/pred/pred_tile_gene.csv" \
  --tcga_csv "${TCGA_CSV}" \
  --slide_id "${SLIDE_ID}" \
  --out_dir "${OUT_ROOT}/eval"

echo ""
echo "=================================================================="
echo "✅ Zero-Shot Pipeline Completed!"
echo "=================================================================="
echo ""
echo "📂 Results:"
echo "   - Predictions: ${OUT_ROOT}/pred/pred_tile_gene.csv"
echo "   - Metrics: ${OUT_ROOT}/eval/metrics_table.csv"
echo "   - Scatter plots: ${OUT_ROOT}/eval/scatter_*.png"
echo ""
echo "📊 Quick metrics preview:"
cat "${OUT_ROOT}/eval/metrics_table.csv" | head -5
echo ""

