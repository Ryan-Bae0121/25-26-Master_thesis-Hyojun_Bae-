#!/usr/bin/env bash
# Zero-shot pipeline using official Loki demo checkpoint (no HF access needed)
# Downloads demo data from Google Drive and uses its checkpoint.pt

set -euo pipefail

echo "=================================================================="
echo "Zero-Shot DEMO LOCAL Pipeline"
echo "Using Official Loki Demo Checkpoint (Foundation Model)"
echo "Slide: TCGA-QK-A6IH-01Z-00-DX1"
echo "=================================================================="
echo ""
echo "[INFO] This uses the demo checkpoint from Loki's basic_usage example"
echo "[INFO] No HuggingFace access required ✅"
echo ""

# Configuration
ROOT=/home/students/hbae/Loki/zero_shot_pipeline
DEMO_DIR=${ROOT}/loki_demo_data
CKPT_PATH=${DEMO_DIR}/data/basic_usage/checkpoint.pt
GDRIVE_FILE_ID="1aPK1nItsOEPxTihUAKMig-vLY-DMMIce"

TILES_DIR=/home/students/hbae/Loki/patches_hdf5_with_annot/TCGA-QK-A6IH-01Z-00-DX1
BANK_H5AD=/home/students/hbae/Loki/hest_hnscc_subset/hnscc_only_2patients/expression_log1p.h5ad
EVAL_SCRIPT=/home/students/hbae/Loki/slide_level_calibrated_metrics.py
TCGA_CSV=/shares/bioit-students/ir_students_2020/TCGA-HNSC/ref_file.csv
SLIDE_ID=TCGA-QK-A6IH-01Z-00-DX1
OUT_ROOT=${ROOT}/zero_shot_QK_A6IH_DEMO_LOCAL

cd "${ROOT}"

echo "Configuration:"
echo "  Tiles: $TILES_DIR"
echo "  Bank: $BANK_H5AD"
echo "  Demo checkpoint: $CKPT_PATH"
echo "  Output: $OUT_ROOT"
echo ""

# Step 0: Download and extract demo data if needed
if [ ! -f "$CKPT_PATH" ]; then
    echo "=================================================================="
    echo "Step 0: Downloading Official Loki Demo Data"
    echo "=================================================================="
    
    mkdir -p "${DEMO_DIR}"
    
    echo "[INFO] Downloading from Google Drive (file_id: ${GDRIVE_FILE_ID})..."
    echo "[INFO] This may take a few minutes..."
    
    gdown "https://drive.google.com/uc?id=${GDRIVE_FILE_ID}" -O "${DEMO_DIR}/loki_demo.zip"
    
    if [ ! -f "${DEMO_DIR}/loki_demo.zip" ]; then
        echo "❌ Download failed!"
        exit 1
    fi
    
    echo "[INFO] Extracting demo data..."
    unzip -q "${DEMO_DIR}/loki_demo.zip" -d "${DEMO_DIR}"
    
    if [ ! -f "$CKPT_PATH" ]; then
        echo "❌ Checkpoint not found after extraction: $CKPT_PATH"
        echo "[INFO] Checking extracted structure..."
        find "${DEMO_DIR}" -name "checkpoint.pt" | head -5
        exit 1
    fi
    
    echo "✅ Demo data downloaded and extracted"
    echo ""
else
    echo "[INFO] ✅ Demo checkpoint already exists: $CKPT_PATH"
    echo ""
fi

# Verify checkpoint exists
if [ ! -f "$CKPT_PATH" ]; then
    echo "❌ Checkpoint not found: $CKPT_PATH"
    exit 1
fi

echo "=================================================================="
echo "Step 1/3: Baseline Run (topk=64, temp=1.0)"
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
  --log1p

# Override with demo checkpoint
echo ""
echo "[INFO] Overriding checkpoint path to demo checkpoint..."
mkdir -p "${OUT_ROOT}/checkpoint"
cp "$CKPT_PATH" "${OUT_ROOT}/checkpoint/checkpoint.pt"

echo ""
echo "✅ Baseline completed!"
echo ""

# Extract paths for sweep
CKPT="${OUT_ROOT}/checkpoint/checkpoint.pt"
BANK_DIR="${OUT_ROOT}/bank"

if [ ! -f "$CKPT" ]; then
    echo "❌ Checkpoint not found: $CKPT"
    exit 1
fi

if [ ! -d "$BANK_DIR" ]; then
    echo "❌ Bank directory not found: $BANK_DIR"
    exit 1
fi

echo "=================================================================="
echo "Step 2/3: Hyperparameter Sweep"
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
    echo "Sweep [$SWEEP_COUNT/$TOTAL_SWEEPS]: topk=$k, temp=$t"
    echo "──────────────────────────────────────────────────────────────"
    
    OUT="${OUT_ROOT}_k${k}_t${t}"
    mkdir -p "${OUT}/pred" "${OUT}/eval"

    # Zero-shot prediction
    echo "  🔮 Running zero-shot prediction..."
    python zero_shot_loki.py \
      --tiles_dir "${TILES_DIR}" \
      --bank_text_emb "npy:${BANK_DIR}/bank_text_emb.npy" \
      --bank_expr "npy:${BANK_DIR}/bank_expr.npy" \
      --bank_genes "${BANK_DIR}/bank_genes.txt" \
      --out_csv "${OUT}/pred/pred_tile_gene.csv" \
      --hf_ckpt "${CKPT}" \
      --device cuda \
      --batch_size 128 \
      --normalize bank_log1p \
      --temp $t \
      --topk $k

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
echo "Step 3/3: Summarizing Sweep Results"
echo "=================================================================="

python - << 'PYEOF'
import os
import glob
import pandas as pd
import numpy as np

print("📊 Collecting DEMO LOCAL sweep results...")

rows = []
for metrics_file in sorted(glob.glob("./zero_shot_QK_A6IH_DEMO_LOCAL_k*/eval/metrics_table.csv")):
    parent_dir = os.path.dirname(os.path.dirname(metrics_file))
    setting_name = os.path.basename(parent_dir).replace("zero_shot_QK_A6IH_DEMO_LOCAL_", "")
    
    try:
        df = pd.read_csv(metrics_file)
        df['hyperparams'] = setting_name
        rows.append(df)
        print(f"   ✅ Loaded: {setting_name}")
    except Exception as e:
        print(f"   ⚠️  Failed to load {metrics_file}: {e}")

if not rows:
    print("⚠️  No sweep results found (baseline only)")
    # Try baseline
    baseline_file = "./zero_shot_QK_A6IH_DEMO_LOCAL/eval/metrics_table.csv"
    if os.path.exists(baseline_file):
        df = pd.read_csv(baseline_file)
        df['hyperparams'] = 'baseline_k64_t1.0'
        rows.append(df)
        print(f"   ✅ Loaded baseline")

if not rows:
    print("❌ No results found!")
    exit(1)

# Combine all results
all_results = pd.concat(rows, ignore_index=True)

print(f"\n✅ Collected {len(rows)} result set(s)")
print(f"   Total rows: {len(all_results)}")

# Save full results
all_results.to_csv("./zero_shot_QK_A6IH_DEMO_LOCAL_sweep_full.csv", index=False)
print(f"\n💾 Saved: ./zero_shot_QK_A6IH_DEMO_LOCAL_sweep_full.csv")

# Create summary
print("\n" + "="*70)
print("Summary: Best Settings by Metric (DEMO LOCAL)")
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
if len(summary) > 0:
    print("\n📈 Top 3 by Pearson (highest):")
    top_n = min(3, len(summary))
    print(summary.nlargest(top_n, 'Pearson')[['hyperparams', 'setting', 'Pearson', 'Spearman', 'MSE']])
    
    if 'sMAPE_zeroAware' in summary.columns:
        print("\n📉 Top 3 by sMAPE_zeroAware (lowest):")
        print(summary.nsmallest(top_n, 'sMAPE_zeroAware')[['hyperparams', 'setting', 'sMAPE_zeroAware', 'Pearson', 'MSE']])
    
    # Find best overall
    print("\n🏆 Best Overall Setting:")
    best_idx = summary['Pearson'].idxmax()
    best = summary.loc[best_idx]
    print(f"   Setting: {best['hyperparams']}")
    print(f"   Pearson: {best['Pearson']:.4f}")
    if 'Spearman' in best:
        print(f"   Spearman: {best['Spearman']:.4f}")
    if 'MSE' in best:
        print(f"   MSE: {best['MSE']:.4f}")

# Save summary
summary.to_csv("./zero_shot_QK_A6IH_DEMO_LOCAL_sweep_summary.csv", index=False)
print(f"\n💾 Saved: ./zero_shot_QK_A6IH_DEMO_LOCAL_sweep_summary.csv")

print("\n" + "="*70)
print("✅ DEMO LOCAL analysis completed!")
print("="*70)

PYEOF

echo ""
echo "=================================================================="
echo "✅ Zero-Shot DEMO LOCAL Pipeline Completed!"
echo "=================================================================="
echo ""
echo "📂 Output Directories:"
echo "   - Baseline: ${OUT_ROOT}/"
echo "   - Sweeps: ${OUT_ROOT}_k*/"
echo ""
echo "📊 Key Files:"
echo "   - Checkpoint: ${CKPT}"
echo "   - Predictions: ${OUT_ROOT}/pred/pred_tile_gene.csv"
echo "   - Metrics: ${OUT_ROOT}/eval/metrics_table.csv"
echo "   - Summary: ./zero_shot_QK_A6IH_DEMO_LOCAL_sweep_summary.csv"
echo ""
echo "📝 Note:"
echo "   ✅ Used official Loki demo checkpoint (foundation model)"
echo "   ✅ No HuggingFace access required"
echo "   ✅ Ready for production once HF access is approved"
echo ""

