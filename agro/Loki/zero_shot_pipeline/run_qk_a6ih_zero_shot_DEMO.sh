#!/usr/bin/env bash
# Zero-shot DEMO pipeline for TCGA-QK-A6IH-01Z-00-DX1
# Uses lightweight demo checkpoint (no HF access needed)

set -euo pipefail

echo "=================================================================="
echo "Zero-Shot DEMO Pipeline (Lightweight Checkpoint)"
echo "Slide: TCGA-QK-A6IH-01Z-00-DX1"
echo "=================================================================="
echo ""
echo "[INFO] Using DEMO mode - no HuggingFace access required"
echo ""

# Configuration
ROOT=/home/students/hbae/Loki/zero_shot_pipeline
TILES_DIR=/home/students/hbae/Loki/patches_hdf5_with_annot/TCGA-QK-A6IH-01Z-00-DX1
BANK_H5AD=/home/students/hbae/Loki/hest_hnscc_subset/hnscc_only_2patients/expression_log1p.h5ad
EVAL_SCRIPT=/home/students/hbae/Loki/slide_level_calibrated_metrics.py
TCGA_CSV=/shares/bioit-students/ir_students_2020/TCGA-HNSC/ref_file.csv
SLIDE_ID=TCGA-QK-A6IH-01Z-00-DX1
OUT_ROOT=${ROOT}/zero_shot_QK_A6IH_DEMO

cd "${ROOT}"

echo "Configuration:"
echo "  Tiles: $TILES_DIR"
echo "  Bank: $BANK_H5AD"
echo "  Output: $OUT_ROOT"
echo ""

echo "=================================================================="
echo "Step 1/3: Baseline Run (DEMO, topk=64, temp=1.0)"
echo "=================================================================="

python run_zero_shot_and_eval.py \
  --tiles_dir "${TILES_DIR}" \
  --bank_h5ad "${BANK_H5AD}" \
  --tcga_csv "${TCGA_CSV}" \
  --slide_id "${SLIDE_ID}" \
  --eval_script "${EVAL_SCRIPT}" \
  --out_root "${OUT_ROOT}" \
  --device cuda \
  --topn 50 \
  --temp 1.0 \
  --topk 64 \
  --batch_size 128 \
  --normalize bank_log1p \
  --agg both \
  --log1p \
  --use_demo

echo ""
echo "✅ Baseline (DEMO) completed!"
echo ""

# Extract bank directory
BANK_DIR="${OUT_ROOT}/bank"

if [ ! -d "$BANK_DIR" ]; then
    echo "❌ Bank directory not found: $BANK_DIR"
    exit 1
fi

echo "=================================================================="
echo "Step 2/3: Hyperparameter Sweep (DEMO)"
echo "=================================================================="
echo "Grid: topk ∈ {32, 64} × temp ∈ {0.5, 1.0}"
echo ""

SWEEP_COUNT=0
TOTAL_SWEEPS=4

for k in 32 64; do
  for t in 0.5 1.0; do
    SWEEP_COUNT=$((SWEEP_COUNT + 1))
    echo ""
    echo "──────────────────────────────────────────────────────────────"
    echo "Sweep [$SWEEP_COUNT/$TOTAL_SWEEPS]: topk=$k, temp=$t (DEMO)"
    echo "──────────────────────────────────────────────────────────────"
    
    OUT="${OUT_ROOT}_k${k}_t${t}"
    mkdir -p "${OUT}/pred" "${OUT}/eval"

    # Zero-shot prediction (DEMO)
    echo "  🔮 Running zero-shot prediction (DEMO)..."
    python zero_shot_loki.py \
      --tiles_dir "${TILES_DIR}" \
      --bank_text_emb "npy:${BANK_DIR}/bank_text_emb.npy" \
      --bank_expr "npy:${BANK_DIR}/bank_expr.npy" \
      --bank_genes "${BANK_DIR}/bank_genes.txt" \
      --out_csv "${OUT}/pred/pred_tile_gene.csv" \
      --device cuda \
      --batch_size 128 \
      --normalize bank_log1p \
      --temp $t \
      --topk $k \
      --use_demo

    # Slide-level evaluation
    echo "  📊 Running slide-level evaluation..."
    python "${EVAL_SCRIPT}" \
      --pred_csv "${OUT}/pred/pred_tile_gene.csv" \
      --tcga_csv "${TCGA_CSV}" \
      --slide_id "${SLIDE_ID}" \
      --out_dir "${OUT}/eval" \
      --agg both \
      --log1p true

    echo "  ✅ Sweep [$SWEEP_COUNT/$TOTAL_SWEEPS] completed: k=$k, t=$t"
  done
done

echo ""
echo "=================================================================="
echo "Step 3/3: Summarizing Sweep Results (DEMO)"
echo "=================================================================="

python - << 'PYEOF'
import os
import glob
import pandas as pd
import numpy as np

print("📊 Collecting DEMO sweep results...")

rows = []
for metrics_file in sorted(glob.glob("./zero_shot_QK_A6IH_DEMO_k*/eval/metrics_table.csv")):
    parent_dir = os.path.dirname(os.path.dirname(metrics_file))
    setting_name = os.path.basename(parent_dir).replace("zero_shot_QK_A6IH_DEMO_", "")
    
    try:
        df = pd.read_csv(metrics_file)
        df['hyperparams'] = setting_name
        rows.append(df)
        print(f"   ✅ Loaded: {setting_name}")
    except Exception as e:
        print(f"   ⚠️  Failed to load {metrics_file}: {e}")

if not rows:
    print("❌ No results found!")
    exit(1)

# Combine all results
all_results = pd.concat(rows, ignore_index=True)

print(f"\n✅ Collected {len(rows)} DEMO sweep results")
print(f"   Total rows: {len(all_results)}")

# Save full results
all_results.to_csv("./zero_shot_QK_A6IH_DEMO_sweep_full.csv", index=False)
print(f"\n💾 Saved: ./zero_shot_QK_A6IH_DEMO_sweep_full.csv")

# Create summary
print("\n" + "="*70)
print("Summary: Best Settings by Metric (DEMO)")
print("="*70)

# Filter for key settings
key_settings = all_results[
    all_results['setting'].str.contains('SUM') & 
    all_results['setting'].str.contains('alphaLS') &
    all_results['setting'].str.contains('log1p')
].copy()

if len(key_settings) == 0:
    print("⚠️  No SUM+alphaLS+log1p settings found, using all results")
    key_settings = all_results.copy()

# Select key columns
metric_cols = [c for c in key_settings.columns if any(x in c for x in 
    ['MSE', 'sMAPE', 'Pearson', 'Spearman'])]

summary_cols = ['hyperparams', 'setting'] + metric_cols
summary = key_settings[summary_cols].copy()

# Sort by different criteria
print("\n📈 Top 3 by Pearson (highest):")
print(summary.nlargest(3, 'Pearson')[['hyperparams', 'setting', 'Pearson', 'Spearman', 'MSE']])

if 'sMAPE_zeroAware' in summary.columns:
    print("\n📉 Top 3 by sMAPE_zeroAware (lowest):")
    print(summary.nsmallest(3, 'sMAPE_zeroAware')[['hyperparams', 'setting', 'sMAPE_zeroAware', 'Pearson', 'MSE']])

# Save summary
summary.to_csv("./zero_shot_QK_A6IH_DEMO_sweep_summary.csv", index=False)
print(f"\n💾 Saved: ./zero_shot_QK_A6IH_DEMO_sweep_summary.csv")

print("\n" + "="*70)
print("✅ DEMO sweep analysis completed!")
print("="*70)

PYEOF

echo ""
echo "=================================================================="
echo "✅ DEMO Pipeline Completed!"
echo "=================================================================="
echo ""
echo "📂 Output Directories:"
echo "   - Baseline: ${OUT_ROOT}/"
echo "   - Sweeps: ${OUT_ROOT}_k*/"
echo ""
echo "📊 Summary Files:"
echo "   - Full results: ./zero_shot_QK_A6IH_DEMO_sweep_full.csv"
echo "   - Summary: ./zero_shot_QK_A6IH_DEMO_sweep_summary.csv"
echo ""
echo "📝 Note: This used DEMO checkpoint for validation."
echo "   For production, use full HF checkpoint (requires access approval)."
echo ""

