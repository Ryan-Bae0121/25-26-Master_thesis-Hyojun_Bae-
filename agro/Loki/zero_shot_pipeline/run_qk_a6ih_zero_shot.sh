#!/bin/bash
# Zero-shot pipeline for TCGA-QK-A6IH-01Z-00-DX1 with hyperparameter sweep

set -e

echo "=================================================================="
echo "Zero-Shot Foundation Model Pipeline"
echo "Slide: TCGA-QK-A6IH-01Z-00-DX1"
echo "=================================================================="

# Configuration
TILES_DIR="/home/students/hbae/Loki/patches_hdf5_with_annot/TCGA-QK-A6IH-01Z-00-DX1"
BANK_H5AD="/home/students/hbae/Loki/hest_hnscc_subset/hnscc_only_2patients/expression_log1p.h5ad"
TCGA_CSV="/shares/bioit-students/ir_students_2020/TCGA-HNSC/ref_file.csv"
SLIDE_ID="TCGA-QK-A6IH-01Z-00-DX1"
EVAL_SCRIPT="/home/students/hbae/Loki/slide_level_calibrated_metrics.py"

echo ""
echo "Configuration:"
echo "  Tiles: $TILES_DIR"
echo "  Bank: $BANK_H5AD"
echo "  TCGA CSV: $TCGA_CSV"
echo "  Slide ID: $SLIDE_ID"
echo ""

# Check HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️  WARNING: HF_TOKEN not set. Checkpoint download may fail."
    echo "   Set it with: export HF_TOKEN=hf_your_token_here"
    echo ""
fi

# Check input files
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
echo "=================================================================="
echo "Step 1/3: Baseline Run (topk=64, temp=1.0)"
echo "=================================================================="

python run_zero_shot_and_eval.py \
  --tiles_dir "$TILES_DIR" \
  --bank_h5ad "$BANK_H5AD" \
  --tcga_csv "$TCGA_CSV" \
  --slide_id "$SLIDE_ID" \
  --eval_script "$EVAL_SCRIPT" \
  --out_root ./zero_shot_QK_A6IH_base \
  --device cuda \
  --topn 50 \
  --temp 1.0 \
  --topk 64 \
  --batch_size 128 \
  --normalize bank_log1p \
  --agg both \
  --log1p

echo ""
echo "✅ Baseline run completed!"
echo ""

# Extract checkpoint and bank paths
CKPT="./zero_shot_QK_A6IH_base/checkpoint/checkpoint.pt"
BANK_DIR="./zero_shot_QK_A6IH_base/bank"

if [ ! -f "$CKPT" ]; then
    echo "❌ Checkpoint not found: $CKPT"
    exit 1
fi

if [ ! -d "$BANK_DIR" ]; then
    echo "❌ Bank directory not found: $BANK_DIR"
    exit 1
fi

echo "=================================================================="
echo "Step 2/3: Hyperparameter Sweep (topk × temp)"
echo "=================================================================="
echo "Grid: topk ∈ {32, 64, 128} × temp ∈ {0.5, 1.0}"
echo ""

SWEEP_COUNT=0
TOTAL_SWEEPS=6

for k in 32 64 128; do
  for t in 0.5 1.0; do
    SWEEP_COUNT=$((SWEEP_COUNT + 1))
    echo ""
    echo "──────────────────────────────────────────────────────────────"
    echo "Sweep [$SWEEP_COUNT/$TOTAL_SWEEPS]: topk=$k, temp=$t"
    echo "──────────────────────────────────────────────────────────────"
    
    OUT="./zero_shot_QK_A6IH_k${k}_t${t}"
    mkdir -p "${OUT}/pred" "${OUT}/eval"

    # Zero-shot prediction
    echo "  🔮 Running zero-shot prediction..."
    python zero_shot_loki.py \
      --tiles_dir "$TILES_DIR" \
      --bank_text_emb "npy:${BANK_DIR}/bank_text_emb.npy" \
      --bank_expr "npy:${BANK_DIR}/bank_expr.npy" \
      --bank_genes "${BANK_DIR}/bank_genes.txt" \
      --out_csv "${OUT}/pred/pred_tile_gene.csv" \
      --hf_ckpt "$CKPT" \
      --device cuda \
      --batch_size 128 \
      --normalize bank_log1p \
      --temp $t \
      --topk $k

    # Slide-level evaluation
    echo "  📊 Running slide-level evaluation..."
    python "$EVAL_SCRIPT" \
      --pred_csv "${OUT}/pred/pred_tile_gene.csv" \
      --tcga_csv "$TCGA_CSV" \
      --slide_id "$SLIDE_ID" \
      --out_dir "${OUT}/eval" \
      --agg both \
      --log1p true

    echo "  ✅ Sweep [$SWEEP_COUNT/$TOTAL_SWEEPS] completed: k=$k, t=$t"
  done
done

echo ""
echo "=================================================================="
echo "Step 3/3: Summarizing Sweep Results"
echo "=================================================================="

python - << 'PYEOF'
import os
import glob
import pandas as pd
import numpy as np

print("📊 Collecting sweep results...")

rows = []
for metrics_file in sorted(glob.glob("./zero_shot_QK_A6IH_k*/eval/metrics_table.csv")):
    # Extract setting from path
    parent_dir = os.path.dirname(os.path.dirname(metrics_file))
    setting_name = os.path.basename(parent_dir).replace("zero_shot_QK_A6IH_", "")
    
    try:
        df = pd.read_csv(metrics_file)
        
        # Add setting identifier
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

print(f"\n✅ Collected {len(rows)} sweep results")
print(f"   Total rows: {len(all_results)}")

# Save full results
all_results.to_csv("./zero_shot_QK_A6IH_sweep_full.csv", index=False)
print(f"\n💾 Saved: ./zero_shot_QK_A6IH_sweep_full.csv")

# Create summary focusing on key metrics
print("\n" + "="*70)
print("Summary: Best Settings by Metric")
print("="*70)

# Filter for most relevant settings (e.g., SUM with alphaLS and log1p)
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
print("\n📈 Top 5 by Pearson (highest):")
print(summary.nlargest(5, 'Pearson')[['hyperparams', 'setting', 'Pearson', 'Spearman', 'MSE']])

if 'sMAPE_zeroAware' in summary.columns:
    print("\n📉 Top 5 by sMAPE_zeroAware (lowest):")
    print(summary.nsmallest(5, 'sMAPE_zeroAware')[['hyperparams', 'setting', 'sMAPE_zeroAware', 'Pearson', 'MSE']])

print("\n📉 Top 5 by MSE (lowest):")
print(summary.nsmallest(5, 'MSE')[['hyperparams', 'setting', 'MSE', 'Pearson', 'Spearman']])

# Save summary
summary.to_csv("./zero_shot_QK_A6IH_sweep_summary.csv", index=False)
print(f"\n💾 Saved: ./zero_shot_QK_A6IH_sweep_summary.csv")

# Find best overall (composite score)
print("\n" + "="*70)
print("Best Overall Setting (Composite Score)")
print("="*70)

# Normalize metrics (0-1 scale)
summary_norm = summary.copy()
for col in metric_cols:
    if 'Pearson' in col or 'Spearman' in col:
        # Higher is better
        summary_norm[col + '_norm'] = (summary_norm[col] - summary_norm[col].min()) / (summary_norm[col].max() - summary_norm[col].min() + 1e-10)
    else:
        # Lower is better (MSE, sMAPE)
        summary_norm[col + '_norm'] = 1 - (summary_norm[col] - summary_norm[col].min()) / (summary_norm[col].max() - summary_norm[col].min() + 1e-10)

# Composite score (equal weights)
norm_cols = [c for c in summary_norm.columns if c.endswith('_norm')]
summary_norm['composite_score'] = summary_norm[norm_cols].mean(axis=1)

best_idx = summary_norm['composite_score'].idxmax()
best_setting = summary_norm.loc[best_idx]

print(f"\n🏆 Best Setting: {best_setting['hyperparams']}")
print(f"   Configuration: {best_setting['setting']}")
print(f"   Composite Score: {best_setting['composite_score']:.4f}")
print(f"\n   Metrics:")
for col in metric_cols:
    if col in best_setting:
        print(f"      {col}: {best_setting[col]:.4f}")

print("\n" + "="*70)
print("✅ Sweep analysis completed!")
print("="*70)

PYEOF

echo ""
echo "=================================================================="
echo "✅ All Steps Completed!"
echo "=================================================================="
echo ""
echo "📂 Output Directories:"
echo "   - Baseline: ./zero_shot_QK_A6IH_base/"
echo "   - Sweeps: ./zero_shot_QK_A6IH_k*/"
echo ""
echo "📊 Summary Files:"
echo "   - Full results: ./zero_shot_QK_A6IH_sweep_full.csv"
echo "   - Summary: ./zero_shot_QK_A6IH_sweep_summary.csv"
echo ""
echo "📈 Next Steps:"
echo "   1. Review metrics in sweep_summary.csv"
echo "   2. Check scatter plots in best setting's eval/ directory"
echo "   3. Compare with Heart fine-tuned model results"
echo ""

