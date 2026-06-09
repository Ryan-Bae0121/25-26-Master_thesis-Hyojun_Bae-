#!/usr/bin/env python3
"""
loki_predex_exact.py
====================
Loki 논문의 정확한 PredEx 방법 구현
- Pretrained OmiCLIP 사용 (파인튜닝 없음)
- 가중 평균(weighted average)으로 유전자 발현 예측

Usage:
    python loki_predex_exact.py \
        --train_csv fold_01_train.csv \
        --val_csv fold_01_val.csv \
        --hvg_file HVG_genelist.txt \
        --output_dir ./loki_predex_results \
        --device cuda:0
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import pearsonr
from tqdm import tqdm


def load_omiclip(checkpoint_path, device):
    """사전 학습된 OmiCLIP 로드 (파인튜닝 없음)"""
    import open_clip
    
    model, _, preprocess = open_clip.create_model_and_transforms(
        'coca_ViT-L-14', pretrained=None)
    
    # Checkpoint 로드
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False)
    
    model = model.to(device).eval()
    return model, preprocess


@torch.no_grad()
def encode_images_batch(model, image_paths, preprocess, device, batch_size=64):
    """배치로 이미지 인코딩"""
    all_embs = []
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Encoding images"):
        batch_paths = image_paths[i:i+batch_size]
        images = []
        
        for path in batch_paths:
            try:
                img = Image.open(path).convert('RGB')
                images.append(preprocess(img))
            except:
                images.append(torch.zeros(3, 224, 224))
        
        img_tensor = torch.stack(images).to(device)
        emb = model.encode_image(img_tensor)
        
        # Handle tuple output
        if isinstance(emb, tuple):
            emb = emb[0]
        
        # Normalize
        emb = F.normalize(emb, dim=-1)
        all_embs.append(emb.cpu())
    
    return torch.cat(all_embs, dim=0)  # (N, 768)


@torch.no_grad()
def encode_gene_sentences(model, gene_lists, tokenizer, device, batch_size=256):
    """유전자 문장 인코딩"""
    all_embs = []
    
    for i in tqdm(range(0, len(gene_lists), batch_size), desc="Encoding genes"):
        batch = gene_lists[i:i+batch_size]
        
        # Gene sentence 만들기 (공백으로 구분)
        sentences = [' '.join(genes) for genes in batch]
        
        tokens = tokenizer(sentences).to(device)
        emb = model.encode_text(tokens)
        
        # Normalize
        emb = F.normalize(emb, dim=-1)
        all_embs.append(emb.cpu())
    
    return torch.cat(all_embs, dim=0)  # (N, 768)


def loki_predex_weighted_average(test_img_emb, train_text_embs, train_exprs,
                                  temperature=0.07, top_k=None, pred_style="exact"):
    """
    Loki PredEx: 가중 평균으로 유전자 발현 예측

    pred_style:
      - "exact": temperature 스케일링 + softmax 가중치 (논문/기본)
      - "case_study": 스케일링 없음, 가중치 = similarity / sum(similarity) (case study 노트북과 동일)
    """
    # 1. 유사도 계산 (cosine similarity)
    similarities = test_img_emb @ train_text_embs.T  # (N,)

    if pred_style == "case_study":
        # Case study: 스케일링 없음, top_k만 적용 가능
        similarities_k = similarities
        train_exprs_k = train_exprs
        if top_k is not None:
            top_indices = similarities.topk(top_k).indices
            similarities_k = similarities[top_indices]
            train_exprs_k = train_exprs[top_indices]
        # 가중치 = similarity / sum(similarity) (softmax 없음)
        s = similarities_k.sum()
        weights = (similarities_k / s) if s > 1e-12 else (torch.ones_like(similarities_k) / similarities_k.numel())
        predicted_expr = (weights[:, None] * train_exprs_k).sum(dim=0)
        return predicted_expr

    # 2. Temperature scaling (exact만)
    similarities = similarities / temperature

    # 3. Top-k 선택 (옵션)
    if top_k is not None:
        top_indices = similarities.topk(top_k).indices
        similarities_k = similarities[top_indices]
        train_exprs_k = train_exprs[top_indices]
    else:
        similarities_k = similarities
        train_exprs_k = train_exprs

    # 4. Softmax로 가중치 계산
    weights = F.softmax(similarities_k, dim=0)

    # 5. 가중 평균
    predicted_expr = (weights[:, None] * train_exprs_k).sum(dim=0)

    return predicted_expr


def main(args):
    import pandas as pd
    import open_clip
    
    device = torch.device(args.device)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    # ─── 1. Load model ─────────────────────────────────────
    print("[1] Loading pretrained OmiCLIP...")
    model, preprocess = load_omiclip(args.pretrained, device)
    tokenizer = open_clip.get_tokenizer('coca_ViT-L-14')
    
    # ─── 2. Load GT ────────────────────────────────────────
    print("[2] Loading ground truth...")
    gt_expr = np.load(args.gt_expr)
    gt_obs = np.load(args.gt_obs, allow_pickle=True)
    all_genes = open(args.gene_list).read().strip().split('\n')
    
    obs_to_idx = {b: i for i, b in enumerate(gt_obs)}
    gene_to_idx = {g: i for i, g in enumerate(all_genes)}
    
    # ─── 3. Load HVG ───────────────────────────────────────
    print("[3] Loading HVG...")
    hvg_genes = open(args.hvg_file).read().strip().split('\n')
    hvg_indices = [gene_to_idx[g] for g in hvg_genes if g in gene_to_idx]
    print(f"  HVG: {len(hvg_indices)}")
    
    # ─── 4. Load datasets ──────────────────────────────────
    print("[4] Loading datasets...")
    
    def extract_obs_key(filepath):
        parts = filepath.split('/')
        gsm = [p for p in parts if p.startswith('GSM') or 
               (len(p) == 7 and p[0].isdigit())]
        sample_id = gsm[0].split('_')[0] if gsm else parts[6].split('_')[0]
        barcode = parts[-1].replace('.png', '')
        return f"{sample_id}_{barcode}_hires"
    
    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)
    
    if 'obs_key' not in train_df.columns:
        train_df['obs_key'] = train_df['filepath'].apply(extract_obs_key)
    if 'obs_key' not in val_df.columns:
        val_df['obs_key'] = val_df['filepath'].apply(extract_obs_key)
    
    # GT 매칭
    train_df = train_df[train_df['obs_key'].isin(obs_to_idx)].reset_index(drop=True)
    val_df = val_df[val_df['obs_key'].isin(obs_to_idx)].reset_index(drop=True)
    
    print(f"  Train: {len(train_df):,} spots")
    print(f"  Val:   {len(val_df):,} spots")
    
    # ─── 5. Encode train set ───────────────────────────────
    print("[5] Encoding train set...")
    
    # Train images → embeddings
    train_img_paths = train_df['filepath'].tolist()
    train_img_embs = encode_images_batch(model, train_img_paths, preprocess, device)
    
    # Train gene sentences → embeddings
    train_gene_lists = []
    train_exprs = []
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Preparing train"):
        genes = row['title'].split()
        train_gene_lists.append(genes)
        
        # GT expression
        spot_idx = obs_to_idx[row['obs_key']]
        expr = gt_expr[spot_idx, hvg_indices]
        train_exprs.append(expr)
    
    train_text_embs = encode_gene_sentences(model, train_gene_lists, tokenizer, device)
    train_exprs = torch.from_numpy(np.array(train_exprs)).float()  # (N, G)
    
    print(f"  Train embeddings: {train_text_embs.shape}")
    print(f"  Train expressions: {train_exprs.shape}")
    
    # ─── 6. Encode val set ─────────────────────────────────
    print("[6] Encoding val set...")
    val_img_paths = val_df['filepath'].tolist()
    val_img_embs = encode_images_batch(model, val_img_paths, preprocess, device)
    
    # Val GT
    val_exprs = []
    for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Preparing val"):
        spot_idx = obs_to_idx[row['obs_key']]
        expr = gt_expr[spot_idx, hvg_indices]
        val_exprs.append(expr)
    val_exprs = np.array(val_exprs)  # (M, G)
    
    # ─── 7. Loki PredEx prediction ─────────────────────────
    pred_style = getattr(args, 'pred_style', 'exact')
    print("[7] Running Loki PredEx (weighted average, style=%s)..." % pred_style)
    
    predictions = []
    for i in tqdm(range(len(val_img_embs)), desc="Predicting"):
        test_emb = val_img_embs[i]
        
        pred_expr = loki_predex_weighted_average(
            test_emb,
            train_text_embs,
            train_exprs,
            temperature=args.temperature,
            top_k=args.top_k,
            pred_style=pred_style,
        )
        predictions.append(pred_expr.numpy())
    
    predictions = np.array(predictions)  # (M, G)
    
    # ─── 8. Evaluation ─────────────────────────────────────
    print("[8] Evaluating...")
    
    # Spot-wise correlation
    spot_corrs = []
    for i in range(len(predictions)):
        if val_exprs[i].std() > 1e-8:
            r, _ = pearsonr(predictions[i], val_exprs[i])
            if np.isfinite(r):
                spot_corrs.append(r)
    
    # Gene-wise correlation
    gene_corrs = []
    for g in range(predictions.shape[1]):
        if val_exprs[:, g].std() > 1e-8:
            r, _ = pearsonr(predictions[:, g], val_exprs[:, g])
            if np.isfinite(r):
                gene_corrs.append(r)
    
    # ─── 9. Results ────────────────────────────────────────
    print("\n" + "="*60)
    print("Loki PredEx Results (Weighted Average)")
    print("="*60)
    print(f"Train spots: {len(train_df):,}")
    print(f"Val spots:   {len(val_df):,}")
    print(f"HVG genes:   {len(hvg_indices)}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-k:       {args.top_k if args.top_k else 'All'}")
    print()
    print(f"Spot-wise Pearson:  mean={np.mean(spot_corrs):.4f}, "
          f"median={np.median(spot_corrs):.4f}, "
          f"std={np.std(spot_corrs):.4f}")
    print(f"Gene-wise Pearson:  mean={np.mean(gene_corrs):.4f}, "
          f"median={np.median(gene_corrs):.4f}, "
          f"std={np.std(gene_corrs):.4f}")
    print("="*60)
    
    # Save
    import json
    pred_style = getattr(args, 'pred_style', 'exact')
    results = {
        'method': 'Loki PredEx (Weighted Average)',
        'pred_style': pred_style,
        'temperature': args.temperature,
        'top_k': args.top_k,
        'train_spots': len(train_df),
        'val_spots': len(val_df),
        'hvg_genes': len(hvg_indices),
        'spot_pearson_mean': float(np.mean(spot_corrs)),
        'spot_pearson_median': float(np.median(spot_corrs)),
        'spot_pearson_std': float(np.std(spot_corrs)),
        'gene_pearson_mean': float(np.mean(gene_corrs)),
        'gene_pearson_median': float(np.median(gene_corrs)),
        'gene_pearson_std': float(np.std(gene_corrs)),
    }
    
    with open(out / 'loki_predex_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # ─── Save predictions if requested ─────────────────────
    if args.save_predictions:
        print("\n[9] Saving predictions...")
        np.save(out / 'predictions.npy', predictions)
        np.save(out / 'ground_truth.npy', val_exprs)
        print(f"  ✓ Predictions saved: {out / 'predictions.npy'}")
        print(f"  ✓ Ground truth saved: {out / 'ground_truth.npy'}")
    
    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", required=True)
    p.add_argument("--val_csv", required=True)
    p.add_argument("--hvg_file", required=True)
    p.add_argument("--gt_expr", default="/project_antwerp/hbae/data/combined_expression_matrix.npy")
    p.add_argument("--gt_obs", default="/project_antwerp/hbae/data/combined_obs.npy")
    p.add_argument("--gene_list", default="/project_antwerp/hbae/data/all_shared_genes.txt")
    p.add_argument("--pretrained", default="/project_antwerp/assets/loki_ckpts/checkpoint.pt")
    p.add_argument("--output_dir", default="./loki_predex_results")
    p.add_argument("--temperature", type=float, default=0.07,
                   help="Temperature for similarity scaling (exact만 사용, case_study는 무시)")
    p.add_argument("--top_k", type=int, default=None,
                   help="Use top-k most similar spots only (None=use all)")
    p.add_argument("--pred_style", choices=["exact", "case_study"], default="exact",
                   help="exact=temp+softmax, case_study=합 정규화만 (스케일링/softmax 없음)")
    p.add_argument("--save_predictions", action='store_true',
                   help="Save prediction and ground truth arrays for gene-level analysis")
    p.add_argument("--device", default="cuda:0")
    
    args = p.parse_args()
    main(args)