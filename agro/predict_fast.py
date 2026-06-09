#!/usr/bin/env python3
"""
predict_fast.py
===============
저장된 embedding으로 빠르게 예측 (~1-2분)
수식/파라미터 실험용

Usage:
    python predict_fast.py \
        --emb_dir /path/to/embeddings/fold_01 \
        --pred_style case_study \
        --top_k 64

pred_style 옵션:
  case_study  : similarity / sum(similarity)  [공식 Loki]
  softmax     : softmax(similarity / temperature)
  img2img     : val image → train IMAGE similarity (비교용)
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from tqdm import tqdm
import json


def predict(val_emb, train_emb, train_exprs, pred_style, temperature, top_k):
    """
    val_emb:      (D,)
    train_emb:    (N, D)
    train_exprs:  (N, G) torch tensor
    """
    sim = val_emb @ train_emb.T  # (N,)

    if top_k is not None:
        idx = sim.topk(top_k).indices
        sim = sim[idx]
        train_exprs = train_exprs[idx]

    if pred_style in ("loki", "case_study", "img2img"):
        # 공식 Loki: weighted_sum / sum(similarity)
        s = sim.sum()
        w = sim / s if s > 1e-12 else torch.ones_like(sim) / sim.numel()
    elif pred_style == "softmax":
        w = F.softmax(sim / temperature, dim=0)
    else:
        raise ValueError(f"Unknown pred_style: {pred_style}")

    return (w[:, None] * train_exprs).sum(dim=0)


def evaluate(predictions, val_exprs):
    spot_corrs, gene_corrs = [], []

    for i in range(len(predictions)):
        if val_exprs[i].std() > 1e-8:
            r, _ = pearsonr(predictions[i], val_exprs[i])
            if np.isfinite(r):
                spot_corrs.append(r)

    for g in range(predictions.shape[1]):
        if val_exprs[:, g].std() > 1e-8:
            r, _ = pearsonr(predictions[:, g], val_exprs[:, g])
            if np.isfinite(r):
                gene_corrs.append(r)

    return np.array(spot_corrs), np.array(gene_corrs)


def main(args):
    emb_dir = Path(args.emb_dir)

    # Load embeddings
    print("Loading embeddings...")
    train_text_embs = torch.tensor(np.load(emb_dir / 'train_text_embs.npy')).float()
    train_img_embs  = torch.tensor(np.load(emb_dir / 'train_img_embs.npy')).float()
    train_exprs     = torch.tensor(np.load(emb_dir / 'train_exprs.npy')).float()
    val_img_embs    = torch.tensor(np.load(emb_dir / 'val_img_embs.npy')).float()
    val_exprs       = np.load(emb_dir / 'val_exprs.npy')

    print(f"  train_text_embs: {train_text_embs.shape}")
    print(f"  train_img_embs:  {train_img_embs.shape}")
    print(f"  val_img_embs:    {val_img_embs.shape}")
    print(f"  val_exprs:       {val_exprs.shape}")

    # val image → train embedding 선택
    if args.pred_style == "img2img":
        train_emb = train_img_embs
        print("\n[Mode] val IMAGE → train IMAGE similarity")
    else:
        train_emb = train_text_embs
        print(f"\n[Mode] val IMAGE → train TEXT similarity ({args.pred_style})")

    # Predict
    predictions = []
    for i in tqdm(range(len(val_img_embs)), desc="Predicting"):
        pred = predict(
            val_img_embs[i], train_emb, train_exprs,
            pred_style=args.pred_style,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        predictions.append(pred.numpy())
    predictions = np.array(predictions)

    # Evaluate
    spot_corrs, gene_corrs = evaluate(predictions, val_exprs)

    print("\n" + "="*60)
    print(f"pred_style:  {args.pred_style}")
    print(f"top_k:       {args.top_k}")
    print(f"temperature: {args.temperature}")
    print("="*60)
    print(f"Spot-wise PCC: mean={spot_corrs.mean():.4f}, median={np.median(spot_corrs):.4f}")
    print(f"Gene-wise PCC: mean={gene_corrs.mean():.4f}, median={np.median(gene_corrs):.4f}")
    print("="*60)

    # Save results
    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        results = {
            'pred_style': args.pred_style,
            'top_k': args.top_k,
            'temperature': args.temperature,
            'spot_pearson_mean':   float(spot_corrs.mean()),
            'spot_pearson_median': float(np.median(spot_corrs)),
            'gene_pearson_mean':   float(gene_corrs.mean()),
            'gene_pearson_median': float(np.median(gene_corrs)),
        }
        with open(out / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        if args.save_predictions:
            np.save(out / 'predictions.npy', predictions)
            np.save(out / 'ground_truth.npy', val_exprs)
        print(f"\nSaved to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--emb_dir",     required=True, help="save_embeddings.py로 저장한 폴더")
    p.add_argument("--pred_style",  default="loki",
                   choices=["loki", "case_study", "softmax", "img2img"])
    p.add_argument("--top_k",       type=int, default=None)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--output_dir",  default=None)
    p.add_argument("--save_predictions", action='store_true')
    args = p.parse_args()
    main(args)