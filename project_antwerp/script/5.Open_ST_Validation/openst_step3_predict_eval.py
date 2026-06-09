#!/usr/bin/env python3
"""
openst_step3_predict_eval.py
============================
predict_per_sample.py 와 동일한 predict() 수식으로 Open-ST external validation 수행.
★ 평가는 Open-ST val set 기준 top-300 expressed genes만 사용 (Loki 논문 방식)

- Query  : Open-ST cell image embeddings  (openst_img_embs.npy)
- Ref    : Training spot image embeddings (train_img_embs.npy) + expressions (train_exprs.npy)
- GT     : Open-ST cell expression        (openst_patches.h5 내 expression)

Usage:
    python openst_step3_predict_eval.py \
        --openst_emb_dir /project_antwerp/hbae/Loki_output/openst_validation/fold_01 \
        --train_emb_dir  /project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new/fold_01 \
        --h5_path        /project_antwerp/hbae/data/Open_ST/openst_patches.h5 \
        --pred_style loki \
        --top_k 50

=== Summary across folds ===
 fold  cellwise_pcc_mean  genewise_pcc_mean  genewise_pcc_median
    1           0.334100           0.035541             0.033229
    2           0.298384           0.004058            -0.006200
    3           0.318362           0.025239             0.015039
    4           0.304492           0.019297             0.011128
    5           0.329749           0.025813             0.014777

Mean Cell-wise PCC: 0.3170 ± 0.0155
Mean Gene-wise PCC: 0.0220 ± 0.0116
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from tqdm import tqdm
import pandas as pd
import h5py


# ── 기존 predict_per_sample.py 와 완전 동일한 함수들 ──────────────────────────

def predict(val_emb, train_emb, train_exprs, pred_style, temperature, top_k, return_debug=False):
    sim = val_emb @ train_emb.T  # (N_train,)

    if top_k is not None:
        idx = sim.topk(top_k).indices
        sim = sim[idx]
        train_exprs = train_exprs[idx]

    if pred_style in ("loki", "case_study", "img2img"):
        s = sim.sum()
        w = sim / s if s.abs() > 1e-12 else torch.ones_like(sim) / sim.numel()
    elif pred_style in ("softmax", "img2img_softmax"):
        w = F.softmax(sim / temperature, dim=0)
    else:
        raise ValueError(f"Unknown pred_style: {pred_style}")

    pred = (w[:, None] * train_exprs).sum(dim=0)

    if return_debug:
        return pred, sim, w
    return pred


def calc_gene_pcc(preds, exprs):
    gene_corrs = []
    for g in range(preds.shape[1]):
        if exprs[:, g].std() > 1e-8:
            r, _ = pearsonr(preds[:, g], exprs[:, g])
            if np.isfinite(r):
                gene_corrs.append(r)
    return np.array(gene_corrs)


def calc_spot_pcc(preds, exprs):
    spot_corrs = []
    for i in range(len(preds)):
        if exprs[i].std() > 1e-8:
            r, _ = pearsonr(preds[i], exprs[i])
            if np.isfinite(r):
                spot_corrs.append(r)
    return np.array(spot_corrs)


def summarize_weights(sim, w):
    w2 = (w * w).sum().item()
    ess = (1.0 / w2) if w2 > 1e-12 else float("inf")
    absw = w.abs()
    return {
        "sim_min":        sim.min().item(),
        "sim_max":        sim.max().item(),
        "sim_mean":       sim.mean().item(),
        "neg_ratio":      (sim < 0).float().mean().item(),
        "sim_sum":        sim.sum().item(),
        "ess":            ess,
        "absw_top1":      absw.max().item(),
        "absw_top5_mass": absw.topk(min(5, absw.numel())).values.sum().item(),
    }


# ── Main ───────────────────────────────────────────────────────────────────

def main(args):
    openst_emb_dir = Path(args.openst_emb_dir)
    train_emb_dir  = Path(args.train_emb_dir)

    # ── 1. Load embeddings ─────────────────────────────────────────────────
    print("Loading embeddings...")
    val_img_embs   = torch.tensor(np.load(openst_emb_dir / 'openst_img_embs.npy')).float()
    train_img_embs = torch.tensor(np.load(train_emb_dir  / 'train_img_embs.npy')).float()
    train_exprs    = torch.tensor(np.load(train_emb_dir  / 'train_exprs.npy')).float()

    print(f"  val_img_embs  (Open-ST): {val_img_embs.shape}")
    print(f"  train_img_embs (Visium): {train_img_embs.shape}")
    print(f"  train_exprs:             {train_exprs.shape}")

    # ── 2. Load Open-ST GT expression from HDF5 ───────────────────────────
    print("\nLoading Open-ST GT expression...")
    with h5py.File(args.h5_path, 'r') as f:
        val_exprs   = f['expression'][:]          # (N_cells, n_genes) float32
        gene_names_bytes = f.attrs['gene_names']
        openst_genes = [g.decode() if isinstance(g, bytes) else g
                        for g in gene_names_bytes]
    print(f"  val_exprs: {val_exprs.shape}")
    print(f"  Open-ST genes: {len(openst_genes)}")

    # ── 3. Gene space alignment ────────────────────────────────────────────
    # train_exprs 는 HVG 2000 genes 순서 (save_embeddings.py 기준)
    # Open-ST 는 1946 genes (HVG 2000 ∩ Open-ST)
    # → 공통 genes만 평가
    hvg_genes = open(args.hvg_file).read().strip().split('\n')
    hvg_genes = [g for g in hvg_genes if g]  # 빈 줄 제거

    # train_exprs 의 gene 순서 = hvg_genes 순서 (save_embeddings.py hvg_indices 기준)
    gene_to_train_idx = {g: i for i, g in enumerate(hvg_genes)}

    # Open-ST ∩ HVG
    shared_genes      = [g for g in openst_genes if g in gene_to_train_idx]
    openst_col_idx    = [openst_genes.index(g) for g in shared_genes]
    train_col_idx     = [gene_to_train_idx[g]  for g in shared_genes]

    val_exprs_shared   = val_exprs[:, openst_col_idx]          # (N_cells, n_shared)
    train_exprs_shared = train_exprs[:, train_col_idx]         # (N_train, n_shared)
    print(f"  Shared genes for evaluation: {len(shared_genes)}")

    # ── 4. Top-300 HEG selection (Loki 논문 방식, val set 기준) ──────────
    mean_expr  = val_exprs_shared.mean(axis=0)
    top300_idx = np.argsort(mean_expr)[::-1][:300].copy()
    print(f"\n★ Top-300 genes selected from Open-ST val set (out of {len(shared_genes)} shared genes)")
    print(f"  Top-5: {[shared_genes[i] for i in top300_idx[:5]]}")

    train_exprs_top300 = train_exprs_shared[:, top300_idx]     # tensor
    val_exprs_top300   = val_exprs_shared[:, top300_idx]       # numpy

    # ── 5. Similarity distribution check (first val cell) ─────────────────
    print("\n[Similarity distribution check - first val cell]")
    train_emb_norm = F.normalize(train_img_embs, dim=-1)
    val_emb_norm   = F.normalize(val_img_embs,   dim=-1)

    sim0 = val_emb_norm[0] @ train_emb_norm.T
    print(f"  min: {sim0.min().item():.4f}")
    print(f"  max: {sim0.max().item():.4f}")
    print(f"  mean: {sim0.mean().item():.4f}")
    print(f"  neg_ratio: {(sim0 < 0).float().mean().item():.4f}")
    print("-" * 50)

    # ── 6. Predict ────────────────────────────────────────────────────────
    print(f"\n[Predicting] pred_style={args.pred_style}, top_k={args.top_k}")
    predictions = []
    for i in tqdm(range(len(val_img_embs)), desc="Predicting"):
        pred = predict(
            val_emb_norm[i], train_emb_norm, train_exprs_top300,
            pred_style=args.pred_style,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        predictions.append(pred.cpu().numpy())
    predictions = np.array(predictions)   # (N_cells, 300)

    # ── 7. Evaluate ───────────────────────────────────────────────────────
    spot_corrs = calc_spot_pcc(predictions, val_exprs_top300)
    gene_corrs = calc_gene_pcc(predictions, val_exprs_top300)

    print("\n" + "=" * 60)
    print(f"Open-ST HNSCC External Validation Results")
    print(f"pred_style : {args.pred_style}")
    print(f"top_k      : {args.top_k}")
    print(f"eval genes : top-300 expressed in Open-ST val set")
    print(f"n_cells    : {len(spot_corrs)} / {len(val_img_embs)}")
    print("=" * 60)
    print(f"Cell-wise (spot-wise) PCC : mean={spot_corrs.mean():.4f}, median={np.median(spot_corrs):.4f}")
    print(f"Gene-wise PCC             : mean={gene_corrs.mean():.4f}, median={np.median(gene_corrs):.4f}")
    print("=" * 60)

    # ── 8. Save ───────────────────────────────────────────────────────────
    out = openst_emb_dir
    np.save(out / 'openst_cellwise_pcc.npy', spot_corrs)
    np.save(out / 'openst_genewise_pcc.npy', gene_corrs)
    np.save(out / 'openst_predictions.npy',  predictions)

    # gene-wise PCC 상위/하위 저장
    gw_df = pd.DataFrame({
        'gene': [shared_genes[i] for i in top300_idx],
        'pcc':  gene_corrs if len(gene_corrs) == 300 else
                [gene_corrs[j] if j < len(gene_corrs) else np.nan for j in range(300)]
    }).sort_values('pcc', ascending=False)
    gw_df.to_csv(out / 'openst_genewise_pcc.csv', index=False)

    print(f"\nTop-10 genes by gene-wise PCC:")
    print(gw_df.head(10).to_string(index=False))
    print(f"\nBottom-10 genes by gene-wise PCC:")
    print(gw_df.tail(10).to_string(index=False))
    print(f"\n✅ Results saved to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--openst_emb_dir", required=True,
                   help="Step 2 output dir (openst_img_embs.npy)")
    p.add_argument("--train_emb_dir",  required=True,
                   help="save_embeddings.py output dir (train_img_embs.npy, train_exprs.npy)")
    p.add_argument("--h5_path",        default="/project_antwerp/hbae/data/Open_ST/openst_patches.h5")
    p.add_argument("--hvg_file",       default="/project_antwerp/hbae/data/0317_hvg_2000_list.txt")
    p.add_argument("--pred_style",     default="loki",
                   choices=["loki", "case_study", "softmax", "img2img", "img2img_softmax"])
    p.add_argument("--top_k",          type=int, default=None)
    p.add_argument("--temperature",    type=float, default=0.07)
    args = p.parse_args()
    main(args)