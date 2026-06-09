#!/bin/bash
# Test script for zero-shot Loki pipeline

set -e

echo "=================================================="
echo "Zero-Shot Loki Pipeline - Test Script"
echo "=================================================="

# Configuration
TILES_DIR="/home/students/hbae/Loki/patches_hdf5_with_annot/TCGA-QK-A6IH-01Z-00-DX1/tiles_224px"
BANK_H5AD="/home/students/hbae/Loki/hest_hnscc_subset/hnscc_only_2patients/expression_log1p.h5ad"
TCGA_CSV="/shares/bioit-students/ir_students_2020/TCGA-HNSC/ref_file.csv"
SLIDE_ID="TCGA-QK-A6IH-01Z-00-DX1"
EVAL_SCRIPT="/home/students/hbae/Loki/slide_level_calibrated_metrics.py"
OUT_ROOT="/home/students/hbae/Loki/zero_shot_results"

echo ""
echo "Configuration:"
echo "  Tiles: $TILES_DIR"
echo "  Bank: $BANK_H5AD"
echo "  TCGA CSV: $TCGA_CSV"
echo "  Slide ID: $SLIDE_ID"
echo "  Output: $OUT_ROOT"
echo ""

# Check if files exist
echo "Checking input files..."
if [ ! -d "$TILES_DIR" ]; then
    echo "❌ Tiles directory not found: $TILES_DIR"
    exit 1
fi
echo "  ✅ Tiles directory exists"

if [ ! -f "$BANK_H5AD" ]; then
    echo "❌ Bank h5ad not found: $BANK_H5AD"
    exit 1
fi
echo "  ✅ Bank h5ad exists"

if [ ! -f "$TCGA_CSV" ]; then
    echo "❌ TCGA CSV not found: $TCGA_CSV"
    exit 1
fi
echo "  ✅ TCGA CSV exists"

if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "❌ Evaluation script not found: $EVAL_SCRIPT"
    exit 1
fi
echo "  ✅ Evaluation script exists"

echo ""
echo "Setting up environment..."
export HF_TOKEN="${HF_TOKEN:-}"
if [ -z "$HF_TOKEN" ]; then
    echo "  ⚠️  HF_TOKEN not set (checkpoint download may fail)"
else
    echo "  ✅ HF_TOKEN is set"
fi

echo ""
echo "=================================================="
echo "Running Zero-Shot Pipeline"
echo "=================================================="

python run_zero_shot_and_eval.py \
  --tiles_dir "$TILES_DIR" \
  --bank_h5ad "$BANK_H5AD" \
  --tcga_csv "$TCGA_CSV" \
  --slide_id "$SLIDE_ID" \
  --eval_script "$EVAL_SCRIPT" \
  --out_root "$OUT_ROOT" \
  --device cuda \
  --topn 50 \
  --temp 1.0 \
  --topk 64 \
  --batch_size 128 \
  --normalize bank_log1p \
  --agg both \
  --log1p

echo ""
echo "=================================================="
echo "✅ Pipeline completed!"
echo "=================================================="
echo ""
echo "Results saved to: $OUT_ROOT"
echo ""
echo "Key files:"
echo "  - Predictions: $OUT_ROOT/pred/pred_tile_gene.csv"
echo "  - Metrics: $OUT_ROOT/eval/metrics_table.csv"
echo "  - Summary: $OUT_ROOT/README.md"
echo ""

