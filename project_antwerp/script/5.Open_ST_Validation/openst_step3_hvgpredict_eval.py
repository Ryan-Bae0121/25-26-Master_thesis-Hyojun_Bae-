#!/usr/bin/env python3
"""
openst_step3_hvgpredict_eval.py
================================
predict_per_sample.py 와 동일한 predict() 수식으로 Open-ST external validation 수행.
저장: /project_antwerp/hbae/Loki_output/openst_validation_hvg300/fold_XX/

Usage:
    python openst_step3_hvgpredict_eval.py \
        --openst_emb_dir /project_antwerp/hbae/Loki_output/openst_validation_agg_v2/fold_01 \
        --train_emb_dir  /project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new/fold_01 \
        --h5_path        /project_antwerp/hbae/data/Open_ST/openst_patches_agg_mc10.h5 \
        --gene_select scanpy_hvg
        최종본
        python openst_step3_hvgpredict_eval.py --openst_emb_dir /project_antwerp/hbae/Loki_output/openst_validation_agg_v2/fold_10 --train_emb_dir /project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new/fold_10 --h5_path /project_antwerp/hbae/data/Open_ST/openst_patches_agg_mc10.h5 --gene_select scanpy_hvg
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


def predict(val_emb, train_emb, train_exprs, pred_style, temperature, top_k, return_debug=False):
    sim = val_emb @ train_emb.T
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


def main(args):
    openst_emb_dir = Path(args.openst_emb_dir)
    train_emb_dir  = Path(args.train_emb_dir)

    # ── 1. Load embeddings ────────────────────────────────────────────────
    print("Loading embeddings...")
    val_img_embs   = torch.tensor(np.load(openst_emb_dir / 'openst_img_embs.npy')).float()
    train_img_embs = torch.tensor(np.load(train_emb_dir  / 'train_img_embs.npy')).float()
    train_exprs    = torch.tensor(np.load(train_emb_dir  / 'train_exprs.npy')).float()
    print(f"  val_img_embs  : {val_img_embs.shape}")
    print(f"  train_img_embs: {train_img_embs.shape}")
    print(f"  train_exprs   : {train_exprs.shape}")

    # ── 2. Load GT expression ─────────────────────────────────────────────
    print("\nLoading Open-ST GT expression...")
    with h5py.File(args.h5_path, 'r') as f:
        val_exprs = f['expression'][:]
        gene_names_bytes = f.attrs['gene_names']
        openst_genes = [g.decode() if isinstance(g, bytes) else g for g in gene_names_bytes]
    print(f"  val_exprs: {val_exprs.shape}")

    # ── 3. Gene space alignment ───────────────────────────────────────────
    hvg_genes = open(args.hvg_file).read().strip().split('\n')
    hvg_genes = [g for g in hvg_genes if g]
    gene_to_train_idx = {g: i for i, g in enumerate(hvg_genes)}
    shared_genes    = [g for g in openst_genes if g in gene_to_train_idx]
    openst_col_idx  = [openst_genes.index(g) for g in shared_genes]
    train_col_idx   = [gene_to_train_idx[g]  for g in shared_genes]
    val_exprs_shared   = val_exprs[:, openst_col_idx]
    train_exprs_shared = train_exprs[:, train_col_idx]
    print(f"  Shared genes: {len(shared_genes)}")

    # ── 4. Top-300 gene selection ─────────────────────────────────────────
    if args.gene_select == 'heg':
        score = val_exprs_shared.mean(axis=0)
        top300_idx = np.argsort(score)[::-1][:300].copy()
        label = "HEG (mean expression, Loki 논문 방식)"
    elif args.gene_select == 'var':
        score = val_exprs_shared.var(axis=0)
        top300_idx = np.argsort(score)[::-1][:300].copy()
        label = "HVG (variance)"
    elif args.gene_select == 'cv':
        mean_e = val_exprs_shared.mean(axis=0)
        std_e  = val_exprs_shared.std(axis=0)
        score  = np.where(mean_e > 0, std_e / mean_e, 0)
        top300_idx = np.argsort(score)[::-1][:300].copy()
        label = "HVG (CV = std/mean)"
    elif args.gene_select == 'scanpy_hvg':
        import scanpy as sc
        adata_tmp = sc.AnnData(X=val_exprs_shared)
        adata_tmp.var_names = shared_genes
        sc.pp.highly_variable_genes(adata_tmp, n_top_genes=300, flavor='seurat')
        hvg_mask = adata_tmp.var['highly_variable'].values
        top300_idx = np.where(hvg_mask)[0].copy()
        label = "HVG (Scanpy seurat flavor)"

    print(f"\n★ Top-300 genes selected by [{label}]")
    print(f"  Selected: {len(top300_idx)} genes")
    print(f"  Top-5: {[shared_genes[i] for i in top300_idx[:5]]}")

    train_exprs_top300 = train_exprs_shared[:, top300_idx]
    val_exprs_top300   = val_exprs_shared[:, top300_idx]

    # ── 5. Predict ────────────────────────────────────────────────────────
    train_emb_norm = F.normalize(train_img_embs, dim=-1)
    val_emb_norm   = F.normalize(val_img_embs,   dim=-1)

    print(f"\n[Predicting] pred_style={args.pred_style}, top_k={args.top_k}")
    predictions = []
    for i in tqdm(range(len(val_img_embs)), desc="Predicting"):
        pred = predict(
            val_emb_norm[i], train_emb_norm, train_exprs_top300,
            pred_style=args.pred_style, temperature=args.temperature, top_k=args.top_k,
        )
        predictions.append(pred.cpu().numpy())
    predictions = np.array(predictions)

    # ── 6. Evaluate ───────────────────────────────────────────────────────
    spot_corrs = calc_spot_pcc(predictions, val_exprs_top300)
    gene_corrs = calc_gene_pcc(predictions, val_exprs_top300)

    print("\n" + "=" * 60)
    print(f"Open-ST HNSCC External Validation Results")
    print(f"gene_select: {args.gene_select}")
    print(f"pred_style : {args.pred_style}, top_k: {args.top_k}")
    print("=" * 60)
    print(f"Cell-wise PCC: mean={spot_corrs.mean():.4f}, median={np.median(spot_corrs):.4f}")
    print(f"Gene-wise PCC: mean={gene_corrs.mean():.4f}, median={np.median(gene_corrs):.4f}")
    print("=" * 60)

    # ── 7. Save (fold별 서브디렉토리) ─────────────────────────────────────
    fold_name = openst_emb_dir.name   # 'fold_01', 'fold_02', ...
    out = Path("/project_antwerp/hbae/Loki_output/openst_validation_hvg300") / fold_name
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / 'openst_cellwise_pcc.npy', spot_corrs)
    np.save(out / 'openst_genewise_pcc.npy', gene_corrs)
    np.save(out / 'openst_predictions.npy',  predictions)

    gw_df = pd.DataFrame({
        'gene': [shared_genes[i] for i in top300_idx],
        'pcc':  gene_corrs if len(gene_corrs) == len(top300_idx) else
                [gene_corrs[j] if j < len(gene_corrs) else np.nan for j in range(len(top300_idx))]
    }).sort_values('pcc', ascending=False)
    gw_df.to_csv(out / 'openst_genewise_pcc.csv', index=False)

    print(f"\nTop-10 genes by gene-wise PCC:")
    print(gw_df.head(10).to_string(index=False))
    print(f"\nBottom-10 genes:")
    print(gw_df.tail(10).to_string(index=False))
    print(f"\n✅ Results saved to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--openst_emb_dir", required=True)
    p.add_argument("--train_emb_dir",  required=True)
    p.add_argument("--h5_path",   default="/project_antwerp/hbae/data/Open_ST/openst_patches_agg_mc10.h5")
    p.add_argument("--hvg_file",  default="/project_antwerp/hbae/data/0317_hvg_2000_list.txt")
    p.add_argument("--pred_style", default="loki",
                   choices=["loki", "case_study", "softmax", "img2img", "img2img_softmax"])
    p.add_argument("--top_k",      type=int, default=None)
    p.add_argument("--gene_select", default="scanpy_hvg",
                   choices=["heg", "var", "cv", "scanpy_hvg"])
    p.add_argument("--temperature", type=float, default=0.07)
    args = p.parse_args()
    main(args)