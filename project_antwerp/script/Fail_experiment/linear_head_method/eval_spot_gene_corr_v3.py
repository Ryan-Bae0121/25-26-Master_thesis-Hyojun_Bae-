#!/usr/bin/env python3
"""
eval_spot_gene_corr_v3.py
=========================
올바른 평가 방법:
- 모든 spot에 동일한 HVG 2089개 전체 적용
- pred: similarity(img_emb, ALL_HVG_emb) → (2089,)
- gt:   gt_expr[spot_idx, ALL_HVG_idx]   → (2089,)
- spot-wise corr: 각 spot에서 pearsonr(pred_2089, gt_2089)
- gene-wise corr: 각 gene에서 pearsonr(pred_all_spots, gt_all_spots)

Usage:
    python eval_spot_gene_corr_v3.py \
        --checkpoint /path/to/epoch_latest.pt \
        --val_csv /path/to/fold_01_val.csv \
        --gene_subset_file /project_antwerp/hbae/HVG_genelist.txt \
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
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────
# Barcode 매칭
# ──────────────────────────────────────────────────────────────
def extract_obs_key(filepath: str) -> str:
    parts = filepath.split('/')
    # GSM... 디렉토리 찾기
    gsm_parts = [p for p in parts if p.startswith('GSM') or 
                 (len(p) == 7 and p[0].isdigit())]  # 17B5776 같은 것도 처리
    
    if gsm_parts:
        sample_dir = gsm_parts[0]
    else:
        sample_dir = parts[6]
    
    sample_id = sample_dir.split('_')[0]
    barcode = parts[-1].replace('.png', '')
    return f"{sample_id}_{barcode}_hires"


# ──────────────────────────────────────────────────────────────
# Model 로드
# ──────────────────────────────────────────────────────────────
def load_model(checkpoint_path: Path, device: str):
    print(f"[1] Loading checkpoint: {checkpoint_path}")
    try:
        import open_clip
    except ImportError:
        sys.exit("❌ open_clip not found.")

    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model, _, preprocess = open_clip.create_model_and_transforms(
        'coca_ViT-L-14', pretrained=None)

    state_dict = ckpt.get('state_dict', ckpt.get('model', ckpt))
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  ✓ Loaded {len(state_dict)} weights "
          f"(missing={len(missing)}, unexpected={len(unexpected)})")
    
    model = model.to(device).eval()
    print(f"  ✓ Model on {device}")
    return model, preprocess


# ──────────────────────────────────────────────────────────────
# GT 로드
# ──────────────────────────────────────────────────────────────
def load_ground_truth(gt_expr_path, gt_obs_path, gene_list_path):
    print("[2] Loading ground truth...")
    expr = np.load(gt_expr_path)
    obs  = np.load(gt_obs_path, allow_pickle=True)
    genes = open(gene_list_path).read().strip().split('\n')

    obs_to_idx  = {b: i for i, b in enumerate(obs)}
    gene_to_idx = {g: i for i, g in enumerate(genes)}

    print(f"  Expression: {expr.shape}, Genes: {len(genes)}")
    return expr, obs_to_idx, gene_to_idx, genes


# ──────────────────────────────────────────────────────────────
# HVG 텍스트 임베딩 (한 번만 계산)
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def compute_text_embeddings(model, hvg_genes: list, tokenizer, device: str,
                            batch_size: int = 256):
    """
    모든 HVG gene의 텍스트 임베딩을 한 번에 계산
    Returns: (n_genes, D) normalized
    """
    print(f"[3] Computing text embeddings for {len(hvg_genes)} HVG genes...")
    all_embs = []
    for i in range(0, len(hvg_genes), batch_size):
        batch = hvg_genes[i:i+batch_size]
        tokens = tokenizer(batch).to(device)
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        all_embs.append(emb.cpu())
    
    text_emb = torch.cat(all_embs, dim=0)  # (n_genes, D)
    print(f"  ✓ Text embeddings: {text_emb.shape}")
    return text_emb


# ──────────────────────────────────────────────────────────────
# Spot inference (배치 처리)
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_spots_batch(model, img_paths: list, preprocess, device: str,
                        text_emb_gpu: torch.Tensor, batch_size: int = 64):
    """
    여러 spot을 배치로 처리
    Returns: (n_spots, n_genes) similarity scores
    """
    all_preds = []
    
    for i in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[i:i+batch_size]
        imgs = []
        valid_mask = []
        
        for p in batch_paths:
            try:
                img = Image.open(p).convert('RGB')
                imgs.append(preprocess(img))
                valid_mask.append(True)
            except:
                imgs.append(torch.zeros(3, 224, 224))
                valid_mask.append(False)
        
        img_tensor = torch.stack(imgs).to(device)  # (B, 3, H, W)
        img_emb = model.encode_image(img_tensor)   # (B, D)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        
        # similarity: (B, n_genes)
        sim = img_emb @ text_emb_gpu.T
        all_preds.append(sim.cpu().numpy())
    
    return np.vstack(all_preds)  # (n_spots, n_genes)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main(args):
    device = args.device
    import open_clip

    # ── 1. Load model ──────────────────────────────────────────
    model, preprocess = load_model(Path(args.checkpoint), device)
    tokenizer = open_clip.get_tokenizer('coca_ViT-L-14')

    # ── 2. Load GT ─────────────────────────────────────────────
    gt_expr, obs_to_idx, gene_to_idx, all_genes = load_ground_truth(
        args.gt_expr, args.gt_obs, args.gene_list)

    # ── 3. Load HVG gene list ──────────────────────────────────
    print(f"[3] Loading HVG gene list: {args.gene_subset_file}")
    hvg_genes = open(args.gene_subset_file).read().strip().split('\n')
    
    # GT에 있는 gene만 사용
    hvg_genes_valid = [g for g in hvg_genes if g in gene_to_idx]
    hvg_indices = [gene_to_idx[g] for g in hvg_genes_valid]
    
    print(f"  HVG total: {len(hvg_genes)}")
    print(f"  HVG in GT: {len(hvg_genes_valid)}")

    # ── 4. Text embeddings (한 번만 계산) ──────────────────────
    text_emb = compute_text_embeddings(
        model, hvg_genes_valid, tokenizer, device)
    text_emb_gpu = text_emb.to(device)

    # ── 5. Load validation CSV ─────────────────────────────────
    print(f"[4] Loading validation CSV...")
    val_df = pd.read_csv(args.val_csv)
    val_df['obs_key'] = val_df['filepath'].apply(extract_obs_key)
    
    # GT 매칭 가능한 spot만
    val_df = val_df[val_df['obs_key'].isin(obs_to_idx)].copy()
    val_df = val_df.reset_index(drop=True)
    print(f"  Spots with GT match: {len(val_df):,}")

    if len(val_df) == 0:
        sys.exit("❌ No spots matched with GT")

    # ── 6. Inference (배치) ────────────────────────────────────
    print(f"[5] Running batch inference...")
    print(f"  Spots: {len(val_df):,}, Genes: {len(hvg_genes_valid)}")
    
    img_paths = val_df['filepath'].tolist()
    
    # 배치 inference
    all_preds = []
    batch_size = args.batch_size
    
    for i in tqdm(range(0, len(img_paths), batch_size), desc="Batches"):
        batch_paths = img_paths[i:i+batch_size]
        imgs = []
        
        for p in batch_paths:
            try:
                img = Image.open(p).convert('RGB')
                imgs.append(preprocess(img))
            except:
                imgs.append(torch.zeros(3, 224, 224))
        
        img_tensor = torch.stack(imgs).to(device)
        
        with torch.no_grad():
            img_emb = model.encode_image(img_tensor)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            sim = (img_emb @ text_emb_gpu.T).cpu().numpy()
        
        all_preds.append(sim)
    
    predictions = np.vstack(all_preds)  # (n_spots, n_hvg)
    print(f"  ✓ Predictions shape: {predictions.shape}")

    # ── 7. Ground truth 수집 ───────────────────────────────────
    print(f"[6] Collecting ground truth...")
    ground_truths = np.zeros((len(val_df), len(hvg_genes_valid)), dtype=np.float32)
    
    for i, row in enumerate(tqdm(val_df.itertuples(), total=len(val_df), desc="GT")):
        spot_idx = obs_to_idx[row.obs_key]
        ground_truths[i] = gt_expr[spot_idx, hvg_indices]
    
    print(f"  ✓ Ground truth shape: {ground_truths.shape}")

    # ── 8. Correlation 계산 ────────────────────────────────────
    print(f"[7] Computing correlations...")

    # Spot-wise (각 spot에서 n_hvg개 gene의 pred vs gt)
    spot_pearson = []
    spot_spearman = []
    
    for i in tqdm(range(len(predictions)), desc="Spot-wise"):
        p = predictions[i]
        g = ground_truths[i]
        
        # GT가 모두 0인 spot은 스킵
        if g.std() < 1e-8:
            spot_pearson.append(np.nan)
            spot_spearman.append(np.nan)
            continue
        
        try:
            r, _ = pearsonr(p, g)
            spot_pearson.append(r if np.isfinite(r) else np.nan)
        except:
            spot_pearson.append(np.nan)
        
        try:
            r, _ = spearmanr(p, g)
            spot_spearman.append(r if np.isfinite(r) else np.nan)
        except:
            spot_spearman.append(np.nan)

    spot_pearson  = np.array(spot_pearson)
    spot_spearman = np.array(spot_spearman)

    # Gene-wise (각 gene에서 n_spots개 spot의 pred vs gt)
    gene_pearson = []
    gene_spearman = []
    
    for g in tqdm(range(len(hvg_genes_valid)), desc="Gene-wise"):
        p_g = predictions[:, g]
        gt_g = ground_truths[:, g]
        
        if gt_g.std() < 1e-8:
            gene_pearson.append(np.nan)
            gene_spearman.append(np.nan)
            continue
        
        try:
            r, _ = pearsonr(p_g, gt_g)
            gene_pearson.append(r if np.isfinite(r) else np.nan)
        except:
            gene_pearson.append(np.nan)
        
        try:
            r, _ = spearmanr(p_g, gt_g)
            gene_spearman.append(r if np.isfinite(r) else np.nan)
        except:
            gene_spearman.append(np.nan)

    gene_pearson  = np.array(gene_pearson)
    gene_spearman = np.array(gene_spearman)

    # ── 9. Results ─────────────────────────────────────────────
    def stats(arr):
        valid = arr[~np.isnan(arr)]
        return {
            "mean":   float(np.mean(valid)),
            "median": float(np.median(valid)),
            "std":    float(np.std(valid)),
            "min":    float(np.min(valid)),
            "max":    float(np.max(valid)),
            "n_valid": int(len(valid)),
            "n_total": int(len(arr)),
        }

    results = {
        "checkpoint": args.checkpoint,
        "val_csv": args.val_csv,
        "n_spots": len(predictions),
        "n_genes": len(hvg_genes_valid),
        "spot_wise_pearson":  stats(spot_pearson),
        "spot_wise_spearman": stats(spot_spearman),
        "gene_wise_pearson":  stats(gene_pearson),
        "gene_wise_spearman": stats(gene_spearman),
    }

    # ── 10. Save ───────────────────────────────────────────────
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "summary.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Per-spot CSV
    pd.DataFrame({
        'obs_key': val_df['obs_key'].values,
        'spot_pearson':  spot_pearson,
        'spot_spearman': spot_spearman,
    }).to_csv(out / "spot_correlations.csv", index=False)

    # Per-gene CSV
    pd.DataFrame({
        'gene': hvg_genes_valid,
        'gene_pearson':  gene_pearson,
        'gene_spearman': gene_spearman,
    }).to_csv(out / "gene_correlations.csv", index=False)

    # ── 11. Print ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"Spots evaluated : {len(predictions):,}")
    print(f"Genes evaluated : {len(hvg_genes_valid):,}")
    print()
    sp = results['spot_wise_pearson']
    ss = results['spot_wise_spearman']
    gp = results['gene_wise_pearson']
    gs = results['gene_wise_spearman']
    print(f"Spot-wise Pearson  : mean={sp['mean']:.4f}, median={sp['median']:.4f}, std={sp['std']:.4f}  (valid={sp['n_valid']:,})")
    print(f"Spot-wise Spearman : mean={ss['mean']:.4f}, median={ss['median']:.4f}, std={ss['std']:.4f}  (valid={ss['n_valid']:,})")
    print()
    print(f"Gene-wise Pearson  : mean={gp['mean']:.4f}, median={gp['median']:.4f}, std={gp['std']:.4f}  (valid={gp['n_valid']:,})")
    print(f"Gene-wise Spearman : mean={gs['mean']:.4f}, median={gs['median']:.4f}, std={gs['std']:.4f}  (valid={gs['n_valid']:,})")
    print()
    print(f"Results saved to: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",       required=True)
    p.add_argument("--val_csv",          required=True)
    p.add_argument("--gene_subset_file", required=True,
                   help="HVG gene list (one per line)")
    p.add_argument("--gt_expr",   default="/project_antwerp/hbae/data/combined_expression_matrix.npy")
    p.add_argument("--gt_obs",    default="/project_antwerp/hbae/data/combined_obs.npy")
    p.add_argument("--gene_list", default="/project_antwerp/hbae/data/all_shared_genes.txt")
    p.add_argument("--output_dir", default="./eval_v3")
    p.add_argument("--device",    default="cuda:0")
    p.add_argument("--batch_size", type=int, default=128,
                   help="Image batch size for inference")
    args = p.parse_args()
    main(args)