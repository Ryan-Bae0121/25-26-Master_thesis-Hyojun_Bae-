#!/bin/bash
#
# 0228_New_HVG_10fold CSV로 save_embeddings.py → predict_per_sample.py를 fold_01 ~ fold_10 순서대로 실행
# zeroshot: 모든 fold에 동일한 pretrained 체크포인트 사용 (PRETRAINED 단일 파일)
#
# Usage:
#   chmod +x run_save_embeddings_0228_10fold.sh
#   ./run_save_embeddings_0228_10fold.sh
#   # 또는 nohup으로 전체 백그라운드:
#   nohup ./run_save_embeddings_0228_10fold.sh > /tmp/save_emb_0228_log.txt 2>&1 &
#

set -e

# 경로 (필요시 수정)
BASE_DIR="/project_antwerp/hbae"
CSV_DIR="${BASE_DIR}/Loki_output/10fold_csv_file/0228_New_HVG_10fold"
OUT_BASE="${BASE_DIR}/Loki_output/embeddings_zeroshot_0228"
GT_EXPR="${BASE_DIR}/data/0228_HVG_NEW/combined_expression_matrix.npy"
GT_OBS="${BASE_DIR}/data/0228_HVG_NEW/combined_obs.npy"
GENE_LIST="${BASE_DIR}/data/0228_HVG_NEW/ST_36s_all_shared_genes.txt"
HVG_FILE="${BASE_DIR}/data/0228_HVG_NEW/0228_HVG_Finetune_gene_list_full.txt"
# pretrained: 모든 fold에 동일 체크포인트 사용 (zeroshot)
PRETRAINED="/project_antwerp/assets/loki_ckpts/checkpoint.pt"
DEVICE="cuda:0"

# 스크립트 위치 (save_embeddings.py와 같은 디렉터리라고 가정; 아니면 절대경로로 바꾸기)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAVE_EMBEDDINGS="${SCRIPT_DIR}/save_embeddings.py"
PREDICT_PER_SAMPLE="${SCRIPT_DIR}/predict_per_sample.py"
if [[ ! -f "$SAVE_EMBEDDINGS" ]]; then
  SAVE_EMBEDDINGS="save_embeddings.py"
fi
if [[ ! -f "$PREDICT_PER_SAMPLE" ]]; then
  PREDICT_PER_SAMPLE="predict_per_sample.py"
fi

mkdir -p "$OUT_BASE"

for f in 01 02 03 04 05 06 07 08 09 10; do
  echo "=============================================="
  echo "Fold ${f}"
  echo "=============================================="
  TRAIN_CSV="${CSV_DIR}/fold_${f}_train.csv"
  VAL_CSV="${CSV_DIR}/fold_${f}_val.csv"
  OUT_DIR="${OUT_BASE}/fold_${f}"

  if [[ ! -f "$TRAIN_CSV" ]] || [[ ! -f "$VAL_CSV" ]]; then
    echo "Skip fold_${f}: CSV not found"
    continue
  fi

  if [[ ! -f "$PRETRAINED" ]]; then
    echo "Skip fold_${f}: checkpoint not found ($PRETRAINED)"
    continue
  fi

  python3 "$SAVE_EMBEDDINGS" \
    --train_csv "$TRAIN_CSV" \
    --val_csv   "$VAL_CSV" \
    --hvg_file  "$HVG_FILE" \
    --gt_expr   "$GT_EXPR" \
    --gt_obs    "$GT_OBS" \
    --gene_list "$GENE_LIST" \
    --pretrained "$PRETRAINED" \
    --output_dir "$OUT_DIR" \
    --device "$DEVICE"

  # save_embeddings 완료 후 predict_per_sample.py 실행
  if [[ -f "$PREDICT_PER_SAMPLE" ]] && [[ -f "$OUT_DIR/train_text_embs.npy" ]]; then
    echo ""
    echo "  [predict_per_sample] Fold ${f}..."
    python3 "$PREDICT_PER_SAMPLE" \
      --emb_dir "$OUT_DIR" \
      --val_csv "$VAL_CSV" \
      --pred_style loki \
      # --top_k 50 || true
  else
    echo "  Skip predict_per_sample (script or embeddings not found)"
  fi
done

echo "=============================================="
echo "Done. Outputs under: $OUT_BASE"
echo "=============================================="
