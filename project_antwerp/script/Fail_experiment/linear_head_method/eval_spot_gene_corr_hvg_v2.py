#!/usr/bin/env python3
"""
eval_spot_gene_corr.py
======================
Fine-tuned Loki 모델로 validation set에서 spot-level gene expression 예측 후
spot-wise / gene-wise correlation 계산

Usage:
    python eval_spot_gene_corr.py \
        --checkpoint /path/to/epoch_latest.pt \
        --val_csv /path/to/fold_01_val.csv \
        --gt_expr /project_antwerp/hbae/data/combined_expression_matrix.npy \
        --gt_obs /project_antwerp/hbae/data/combined_obs.npy \
        --gene_list /project_antwerp/hbae/data/all_shared_genes.txt \
        --output_dir ./eval_results \
        --device cuda:0
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import pearsonr
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────
# Barcode 매칭
# ──────────────────────────────────────────────────────────────
def extract_obs_key(filepath: str) -> str:
    """
    filepath: /project_antwerp/.../GSM6339631_s1/patches/AAACACCAATAACTGC-1.png
    → GSM6339631_AAACACCAATAACTGC-1_hires
    """
    parts = filepath.split('/')
    sample_dir = parts[6]  # GSM6339631_s1
    sample_id = sample_dir.split('_')[0]  # GSM6339631
    barcode = parts[8].replace('.png', '')  # AAACACCAATAACTGC-1
    
    obs_key = f"{sample_id}_{barcode}_hires"
    return obs_key


# ──────────────────────────────────────────────────────────────
# Checkpoint 로드
# ──────────────────────────────────────────────────────────────
def load_model(checkpoint_path: Path, device: str):
    """Loki checkpoint 로드"""
    print(f"[1] Loading checkpoint: {checkpoint_path}")
    
    # open_clip 임포트
    try:
        import open_clip
    except ImportError:
        sys.exit("❌ open_clip not found. Install: pip install open-clip-torch")
    
    # checkpoint 로드
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    # model 구조 생성 (pretrained 명시적으로 None)
    model_name = "coca_ViT-L-14"
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=None,  # 중요: pretrained weights 안 씀
    )
    
    # state_dict 추출
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
        print(f"  Loaded from 'state_dict' key (epoch {ckpt.get('epoch', 'unknown')})")
    elif 'model' in ckpt:
        state_dict = ckpt['model']
        print(f"  Loaded from 'model' key")
    else:
        state_dict = ckpt
        print(f"  Direct state dict")
    
    # 'module.' prefix 제거 (DDP 학습 시)
    state_dict_clean = {}
    for k, v in state_dict.items():
        k_clean = k.replace('module.', '')
        state_dict_clean[k_clean] = v
    
    # Load state dict
    missing_keys, unexpected_keys = model.load_state_dict(state_dict_clean, strict=False)
    
    # 로딩 결과 출력
    print(f"  ✓ Loaded {len(state_dict_clean)} weights")
    if missing_keys:
        print(f"  ⚠️  Missing keys: {len(missing_keys)} (first 5: {missing_keys[:5]})")
    if unexpected_keys:
        print(f"  ⚠️  Unexpected keys: {len(unexpected_keys)} (first 5: {unexpected_keys[:5]})")
    
    model = model.to(device)
    model.eval()
    
    print(f"  ✓ Model loaded to {device}")
    return model, preprocess


# ──────────────────────────────────────────────────────────────
# GT 로드
# ──────────────────────────────────────────────────────────────
def load_ground_truth(gt_expr_path: Path, gt_obs_path: Path, gene_list_path: Path):
    """GT expression matrix + obs + gene list 로드"""
    print("[2] Loading ground truth...")
    
    expr = np.load(gt_expr_path)  # (154763, 12009)
    obs = np.load(gt_obs_path, allow_pickle=True)  # (154763,)
    genes = open(gene_list_path).read().strip().split('\n')  # 12009 genes
    
    print(f"  Expression shape: {expr.shape}")
    print(f"  Obs shape: {obs.shape}")
    print(f"  Genes: {len(genes)}")
    
    # obs를 dict로 변환 (빠른 검색)
    obs_to_idx = {barcode: i for i, barcode in enumerate(obs)}
    
    # gene symbol → index
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    
    return expr, obs_to_idx, gene_to_idx, genes


# ──────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_spot(model, img_path: str, gene_symbols: list, preprocess, tokenizer, device: str):
    """
    단일 spot 이미지에서 gene expression 예측
    
    Returns:
        pred_expr: (n_genes,) 예측 발현값
    """
    # 이미지 로드
    img = Image.open(img_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0).to(device)  # (1, 3, H, W)
    
    # 이미지 임베딩
    img_emb = model.encode_image(img_tensor)  # (1, D)
    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
    
    # 텍스트 임베딩 (gene symbols)
    texts = tokenizer(gene_symbols).to(device)  # (n_genes, context_len)
    text_emb = model.encode_text(texts)  # (n_genes, D)
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    
    # Similarity (cosine)
    similarity = (img_emb @ text_emb.T).squeeze(0)  # (n_genes,)
    
    # Temperature scaling (similarity를 그대로 사용, softmax 제거)
    tau = 0.07
    pred_expr = (similarity / tau).cpu().numpy()
    
    return pred_expr


# ──────────────────────────────────────────────────────────────
# Main evaluation
# ──────────────────────────────────────────────────────────────
def main(args):
    device = args.device
    
    # ── 1. Load model ──────────────────────────────────────────
    model, preprocess = load_model(Path(args.checkpoint), device)
    
    # Tokenizer
    import open_clip
    tokenizer = open_clip.get_tokenizer("coca_ViT-L-14")
    
    # ── 2. Load GT ─────────────────────────────────────────────
    gt_expr, obs_to_idx, gene_to_idx, all_genes = load_ground_truth(
        Path(args.gt_expr),
        Path(args.gt_obs),
        Path(args.gene_list),
    )
    
    # ── 2b. Load gene subset (HVG) if specified ────────────────
    hvg_set = None
    if args.gene_subset_file:
        print(f"[2b] Loading gene subset: {args.gene_subset_file}")
        hvg_genes = open(args.gene_subset_file).read().strip().split('\n')
        hvg_set = set(hvg_genes)
        print(f"  Gene subset size: {len(hvg_set):,}")
        
        # Check overlap with GT genes
        overlap = hvg_set & set(all_genes)
        print(f"  Overlap with GT: {len(overlap):,} / {len(hvg_set):,} ({100*len(overlap)/len(hvg_set):.1f}%)")
        
        if len(overlap) == 0:
            sys.exit("❌ No overlap between gene subset and GT genes")
    
    # ── 3. Load validation CSV ─────────────────────────────────
    print("[3] Loading validation CSV...")
    val_df = pd.read_csv(args.val_csv)
    val_df['obs_key'] = val_df['filepath'].apply(extract_obs_key)
    val_df['genes'] = val_df['title'].apply(lambda x: x.split())
    
    print(f"  Total validation spots: {len(val_df):,}")
    
    # GT와 매칭 가능한 spot만 필터링
    val_df = val_df[val_df['obs_key'].isin(obs_to_idx.keys())].copy()
    print(f"  Spots with GT: {len(val_df):,}")
    
    if len(val_df) == 0:
        sys.exit("❌ No validation spots matched with GT")
    
    # ── 4. Inference ───────────────────────────────────────────
    print("[4] Running inference...")
    
    predictions = []
    ground_truths = []
    valid_spots = []
    
    for idx, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Predicting"):
        img_path = row['filepath']
        gene_symbols = row['genes']  # 50 HVG
        obs_key = row['obs_key']
        
        # HVG filtering (if gene_subset_file provided)
        if hvg_set is not None:
            gene_symbols = [g for g in gene_symbols if g in hvg_set]
            if len(gene_symbols) == 0:
                continue
        
        # 이미지 파일 존재 확인
        if not Path(img_path).exists():
            continue
        
        try:
            # Prediction
            pred_expr = predict_spot(model, img_path, gene_symbols, preprocess, tokenizer, device)
            
            # GT 추출 (해당 spot의 50 HVG만)
            spot_idx = obs_to_idx[obs_key]
            gene_indices = [gene_to_idx[g] for g in gene_symbols if g in gene_to_idx]
            
            if len(gene_indices) == 0:
                continue
            
            gt_expr_spot = gt_expr[spot_idx, gene_indices]
            pred_expr_spot = pred_expr[:len(gene_indices)]  # gene 개수 맞추기
            
            predictions.append(pred_expr_spot)
            ground_truths.append(gt_expr_spot)
            valid_spots.append(obs_key)
            
        except Exception as e:
            print(f"  ⚠️  Error at {obs_key}: {e}")
            continue
    
    print(f"  ✓ Successfully predicted: {len(predictions):,} spots")
    
    # ── 5. Correlation 계산 ────────────────────────────────────
    print("[5] Computing correlations...")
    
    predictions = np.array(predictions)  # (n_spots, 50)
    ground_truths = np.array(ground_truths)  # (n_spots, 50)
    
    # (A) Spot-wise correlation
    spot_corrs = []
    for i in range(len(predictions)):
        try:
            corr, _ = pearsonr(predictions[i], ground_truths[i])
            spot_corrs.append(corr if np.isfinite(corr) else 0.0)
        except:
            spot_corrs.append(0.0)
    
    spot_corrs = np.array(spot_corrs)
    
    # (B) Gene-wise correlation
    gene_corrs = []
    for g in range(predictions.shape[1]):
        try:
            corr, _ = pearsonr(predictions[:, g], ground_truths[:, g])
            gene_corrs.append(corr if np.isfinite(corr) else 0.0)
        except:
            gene_corrs.append(0.0)
    
    gene_corrs = np.array(gene_corrs)
    
    # ── 6. Results ─────────────────────────────────────────────
    results = {
        "checkpoint": str(args.checkpoint),
        "n_spots_evaluated": len(predictions),
        "n_genes_per_spot": predictions.shape[1],
        "spot_wise_corr": {
            "mean": float(spot_corrs.mean()),
            "median": float(np.median(spot_corrs)),
            "std": float(spot_corrs.std()),
            "min": float(spot_corrs.min()),
            "max": float(spot_corrs.max()),
        },
        "gene_wise_corr": {
            "mean": float(gene_corrs.mean()),
            "median": float(np.median(gene_corrs)),
            "std": float(gene_corrs.std()),
            "min": float(gene_corrs.min()),
            "max": float(gene_corrs.max()),
        },
    }
    
    # ── 7. Save ────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON summary
    json_path = output_dir / "correlation_summary.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Per-spot CSV
    spot_df = pd.DataFrame({
        'obs_key': valid_spots,
        'spot_corr': spot_corrs,
    })
    spot_csv = output_dir / "spot_correlations.csv"
    spot_df.to_csv(spot_csv, index=False)
    
    # Per-gene CSV
    gene_df = pd.DataFrame({
        'gene_idx': range(len(gene_corrs)),
        'gene_corr': gene_corrs,
    })
    gene_csv = output_dir / "gene_correlations.csv"
    gene_df.to_csv(gene_csv, index=False)
    
    # ── 8. Print ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"Spots evaluated: {len(predictions):,}")
    print(f"Genes per spot: {predictions.shape[1]}")
    print()
    print("Spot-wise correlation:")
    print(f"  Mean   : {results['spot_wise_corr']['mean']:.4f}")
    print(f"  Median : {results['spot_wise_corr']['median']:.4f}")
    print(f"  Std    : {results['spot_wise_corr']['std']:.4f}")
    print()
    print("Gene-wise correlation:")
    print(f"  Mean   : {results['gene_wise_corr']['mean']:.4f}")
    print(f"  Median : {results['gene_wise_corr']['median']:.4f}")
    print(f"  Std    : {results['gene_wise_corr']['std']:.4f}")
    print()
    print(f"Results saved to: {output_dir}")
    print(f"  - {json_path.name}")
    print(f"  - {spot_csv.name}")
    print(f"  - {gene_csv.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate spot-level gene expression prediction")
    
    parser.add_argument("--checkpoint", required=True, help="Fine-tuned model checkpoint (.pt)")
    parser.add_argument("--val_csv", required=True, help="Validation CSV (filepath, title)")
    parser.add_argument(
        "--gt_expr",
        default="/project_antwerp/hbae/data/combined_expression_matrix.npy",
        help="Ground truth expression matrix",
    )
    parser.add_argument(
        "--gt_obs",
        default="/project_antwerp/hbae/data/combined_obs.npy",
        help="Ground truth observation metadata",
    )
    parser.add_argument(
        "--gene_list",
        default="/project_antwerp/hbae/data/all_shared_genes.txt",
        help="Gene list (one per line)",
    )
    parser.add_argument("--output_dir", default="./eval_results", help="Output directory")
    parser.add_argument("--device", default="cuda:0", help="Device (cuda:0, cpu)")
    parser.add_argument(
        "--gene_subset_file",
        default=None,
        help="Optional: Only evaluate on genes in this file (one gene per line, e.g., HVG_genelist.txt)",
    )
    
    args = parser.parse_args()
    main(args)