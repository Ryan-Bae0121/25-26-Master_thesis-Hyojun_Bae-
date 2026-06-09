#!/usr/bin/env python3
"""
openst_step3_toppcc_eval.py
===========================
전체 1946개 gene으로 배치 예측 후 gene-wise PCC 높은 top-300 선택하여 재평가.
(기존 cell-by-cell loop 대신 배치 처리로 속도 개선)

Usage:
    python openst_step3_toppcc_eval.py \
        --openst_emb_dir /project_antwerp/hbae/Loki_output/openst_validation_agg_v2/fold_01 \
        --train_emb_dir  /project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new/fold_01 \
        --h5_path        /project_antwerp/hbae/data/Open_ST/openst_patches_agg_mc10.h5
        이게 맞음 01 ~10

        python openst_step3_predict_top_pcc_eval.py         --openst_emb_dir /project_antwerp/hbae/Loki_output/openst_validation_agg_v2/fold_10         --train_emb_dir  /project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new/fold_10
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

parser = argparse.ArgumentParser()
parser.add_argument("--openst_emb_dir", required=True)
parser.add_argument("--train_emb_dir",  required=True)
parser.add_argument("--h5_path", default="/project_antwerp/hbae/data/Open_ST/openst_patches_agg_mc10.h5")
parser.add_argument("--hvg_file", default="/project_antwerp/hbae/data/0317_hvg_2000_list.txt")
parser.add_argument("--pred_style", default="loki")
parser.add_argument("--top_k",      type=int, default=None)
parser.add_argument("--temperature", type=float, default=0.07)
parser.add_argument("--chunk_size", type=int, default=500,
                    help="배치 크기 (메모리 조정용, default=500)")
args = parser.parse_args()

openst_emb_dir = Path(args.openst_emb_dir)
train_emb_dir  = Path(args.train_emb_dir)

# ── 1. Load embeddings ─────────────────────────────────────────────────────
print("Loading embeddings...")
val_img_embs   = torch.tensor(np.load(openst_emb_dir / 'openst_img_embs.npy')).float()
train_img_embs = torch.tensor(np.load(train_emb_dir  / 'train_img_embs.npy')).float()
train_exprs    = torch.tensor(np.load(train_emb_dir  / 'train_exprs.npy')).float()

print(f"  val_img_embs  : {val_img_embs.shape}")
print(f"  train_img_embs: {train_img_embs.shape}")
print(f"  train_exprs   : {train_exprs.shape}")

# ── 2. Load GT expression ──────────────────────────────────────────────────
print("\nLoading GT expression...")
with h5py.File(args.h5_path, 'r') as f:
    val_exprs = f['expression'][:]
    gene_names_bytes = f.attrs['gene_names']
    openst_genes = [g.decode() if isinstance(g, bytes) else g for g in gene_names_bytes]
print(f"  val_exprs: {val_exprs.shape}")

# ── 3. Gene space alignment ────────────────────────────────────────────────
hvg_genes = open(args.hvg_file).read().strip().split('\n')
hvg_genes  = [g for g in hvg_genes if g]
gene_to_train_idx = {g: i for i, g in enumerate(hvg_genes)}

shared_genes   = [g for g in openst_genes if g in gene_to_train_idx]
openst_col_idx = [openst_genes.index(g) for g in shared_genes]
train_col_idx  = [gene_to_train_idx[g]  for g in shared_genes]

val_exprs_shared   = val_exprs[:, openst_col_idx]
train_exprs_shared = train_exprs[:, train_col_idx]
print(f"  Shared genes: {len(shared_genes)}")

# ── 4. Normalize embeddings ────────────────────────────────────────────────
train_emb_norm = F.normalize(train_img_embs, dim=-1)
val_emb_norm   = F.normalize(val_img_embs,   dim=-1)

# ── 5. 배치 예측 (전체 shared genes) ──────────────────────────────────────
print(f"\n[Phase 1] 전체 {len(shared_genes)}개 gene으로 배치 예측 (chunk_size={args.chunk_size}) ...")

N_val   = val_emb_norm.shape[0]
N_train = train_emb_norm.shape[0]
predictions_all = np.zeros((N_val, len(shared_genes)), dtype=np.float32)

for start in tqdm(range(0, N_val, args.chunk_size), desc="Predicting (batch)"):
    end   = min(start + args.chunk_size, N_val)
    chunk = val_emb_norm[start:end]            # (B, D)

    # cosine similarity
    sim = chunk @ train_emb_norm.T             # (B, N_train)

    # top_k filtering
    if args.top_k is not None:
        topk_vals, topk_idx = sim.topk(args.top_k, dim=1)
        # sparse weighted average
        for i in range(end - start):
            s = topk_vals[i]
            e = train_exprs_shared[topk_idx[i]]  # (K, n_genes)
            if args.pred_style == 'loki':
                s_sum = s.sum()
                w = s / s_sum if s_sum.abs() > 1e-12 else torch.ones_like(s) / s.numel()
            else:
                w = F.softmax(s / args.temperature, dim=0)
            predictions_all[start + i] = (w.unsqueeze(0) @ e.float()).squeeze(0).cpu().numpy()
    else:
        # 전체 train spot 사용
        if args.pred_style == 'loki':
            s_sum = sim.sum(dim=1, keepdim=True)
            w = sim / s_sum.clamp(min=1e-12)   # (B, N_train)
        else:
            w = F.softmax(sim / args.temperature, dim=1)

        pred = w @ train_exprs_shared.float()  # (B, n_genes)
        predictions_all[start:end] = pred.cpu().numpy()

print(f"  Predictions shape: {predictions_all.shape}")

# ── 6. Gene-wise PCC (전체 gene) ──────────────────────────────────────────
print("\n[Phase 2] 전체 gene-wise PCC 계산 ...")
gene_pccs = np.full(len(shared_genes), -999.0)

for j in tqdm(range(len(shared_genes)), desc="Gene-wise PCC"):
    gt   = val_exprs_shared[:, j]
    pred = predictions_all[:, j]
    if gt.std() > 1e-8 and pred.std() > 1e-8:
        r, _ = pearsonr(gt, pred)
        if np.isfinite(r):
            gene_pccs[j] = r

valid_mask = gene_pccs > -999
print(f"  Valid genes: {valid_mask.sum()} / {len(shared_genes)}")
print(f"  Overall gene-wise PCC mean: {gene_pccs[valid_mask].mean():.4f}")

# ── 7. Top-300 PCC gene 선택 ──────────────────────────────────────────────
top300_idx = np.argsort(gene_pccs)[::-1][:300].copy()
top300_genes = [shared_genes[i] for i in top300_idx]
top300_pccs  = gene_pccs[top300_idx]

print(f"\n★ Top-300 PCC genes selected")
print(f"  PCC range: {top300_pccs.min():.4f} ~ {top300_pccs.max():.4f}")
print(f"  Top-10 genes: {top300_genes[:10]}")

# ── 8. Top-300 재평가 ─────────────────────────────────────────────────────
print("\n[Phase 3] Top-300 PCC genes로 재평가 ...")
preds_top300 = predictions_all[:, top300_idx]
gt_top300    = val_exprs_shared[:, top300_idx]

# Cell-wise PCC
cellwise_pcc = []
for i in range(N_val):
    if gt_top300[i].std() > 1e-8 and preds_top300[i].std() > 1e-8:
        r, _ = pearsonr(gt_top300[i], preds_top300[i])
        if np.isfinite(r):
            cellwise_pcc.append(r)
cellwise_pcc = np.array(cellwise_pcc)

# Gene-wise PCC (top-300만)
genewise_pcc = top300_pccs.copy()

print(f"\n{'='*60}")
print(f"Open-ST top_pcc Evaluation Results")
print(f"pred_style: {args.pred_style}, top_k: {args.top_k}")
print(f"{'='*60}")
print(f"Cell-wise PCC (top-300 PCC genes): mean={cellwise_pcc.mean():.4f}, median={np.median(cellwise_pcc):.4f}")
print(f"Gene-wise PCC (top-300 PCC genes): mean={genewise_pcc.mean():.4f}, median={np.median(genewise_pcc):.4f}")
print(f"{'='*60}")

print(f"\nTop-10 genes by PCC:")
for gene, pcc in zip(top300_genes[:10], top300_pccs[:10]):
    print(f"  {gene:20s}: {pcc:.4f}")

# ── 9. Save ────────────────────────────────────────────────────────────────
fold_name = openst_emb_dir.name   # 'fold_01', 'fold_02', ...
out = Path("/project_antwerp/hbae/Loki_output/openst_validation_TopPCC300") / fold_name
out.mkdir(parents=True, exist_ok=True)
np.save(out / 'openst_cellwise_pcc_toppcc.npy', cellwise_pcc)
np.save(out / 'openst_genewise_pcc_toppcc.npy', genewise_pcc)
np.save(out / 'openst_allgene_pcc.npy', gene_pccs)  # 전체 gene PCC 저장

gw_df = pd.DataFrame({'gene': top300_genes, 'pcc': top300_pccs})
gw_df.to_csv(out / 'openst_genewise_pcc_toppcc.csv', index=False)

print(f"\n✅ Saved to {out}")
print(f"  openst_cellwise_pcc_toppcc.npy")
print(f"  openst_genewise_pcc_toppcc.npy")
print(f"  openst_allgene_pcc.npy  (전체 {len(shared_genes)}개 gene PCC)")
print(f"  openst_genewise_pcc_toppcc.csv")