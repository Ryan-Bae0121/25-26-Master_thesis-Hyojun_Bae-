#!/usr/bin/env python3
"""
predict_visium_hd_v2.py
=======================
SEQUOIA 논문 방식 적용:
1. Median filtering (3×3 공간 노이즈 제거)
2. Unique expression values 필터 (10개 미만 제외)
3. Percentile 정규화 (0~100)

Usage:
    python predict_visium_hd_v2.py \
        --train_emb_dir /project_antwerp/hbae/Loki_output/0228_epoch10_finetune_embedding/fold_07 \
        --visium_hd_dir /project_antwerp/hbae/Loki_output/visium_hd_embeddings/fold_07 \
        --pred_style loki \
        --top_k 500 \
        --device cuda:0
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from scipy.ndimage import median_filter
from tqdm import tqdm


def predict_batch(val_embs, train_emb, train_exprs, pred_style, temperature, top_k, device):
    val_embs    = val_embs.to(device)
    train_emb   = train_emb.to(device)
    train_exprs = train_exprs.to(device)
    sim_matrix  = val_embs @ train_emb.T

    preds = []
    for sim in sim_matrix:
        if top_k is not None:
            idx    = sim.topk(min(top_k, sim.shape[0])).indices
            sim_k  = sim[idx]
            expr_k = train_exprs[idx]
        else:
            sim_k  = sim
            expr_k = train_exprs

        if pred_style == 'loki':
            s = sim_k.sum()
            w = sim_k / s if s.abs() > 1e-12 else torch.ones_like(sim_k) / sim_k.numel()
        elif pred_style == 'softmax':
            w = F.softmax(sim_k / temperature, dim=0)

        preds.append((w[:, None] * expr_k).sum(dim=0).cpu())
    return torch.stack(preds).numpy()


def to_percentile(arr):
    """각 유전자를 0~100 percentile로 정규화"""
    result = np.zeros_like(arr, dtype=np.float32)
    for g in range(arr.shape[1]):
        col = arr[:, g]
        ranks = col.argsort().argsort().astype(np.float32)
        result[:, g] = ranks / (len(ranks) - 1) * 100 if len(ranks) > 1 else 0
    return result


def apply_spatial_median_filter(exprs, coords, kernel_size=3):
    """
    2D 공간 grid에서 median filter 적용
    coords: (N, 2) array of (col, row)
    """
    col_coords = coords[:, 0]
    row_coords = coords[:, 1]

    # 고유한 col, row 값
    unique_cols = np.unique(col_coords)
    unique_rows = np.unique(row_coords)

    col_to_idx = {c: i for i, c in enumerate(unique_cols)}
    row_to_idx = {r: i for i, r in enumerate(unique_rows)}

    n_cols = len(unique_cols)
    n_rows = len(unique_rows)
    n_genes = exprs.shape[1]

    # 2D grid 만들기
    grid = np.zeros((n_rows, n_cols, n_genes), dtype=np.float32)
    valid = np.zeros((n_rows, n_cols), dtype=bool)

    for i, (col, row) in enumerate(coords):
        ci = col_to_idx[col]
        ri = row_to_idx[row]
        grid[ri, ci] = exprs[i]
        valid[ri, ci] = True

    # 각 유전자에 median filter 적용
    filtered_grid = np.zeros_like(grid)
    for g in range(n_genes):
        filtered_grid[:, :, g] = median_filter(grid[:, :, g], size=kernel_size)

    # 다시 1D로 변환
    filtered_exprs = np.zeros_like(exprs)
    for i, (col, row) in enumerate(coords):
        ci = col_to_idx[col]
        ri = row_to_idx[row]
        filtered_exprs[i] = filtered_grid[ri, ci]

    return filtered_exprs


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
    train_dir = Path(args.train_emb_dir)
    hd_dir    = Path(args.visium_hd_dir)
    device    = torch.device(args.device)

    # ── [1] train 임베딩 로드
    print('\n[1] train 임베딩 로드')
    train_text_embs = torch.tensor(np.load(train_dir / 'train_text_embs.npy')).float()
    train_exprs     = torch.tensor(np.load(train_dir / 'train_exprs.npy')).float()
    train_emb_norm  = F.normalize(train_text_embs, dim=-1)
    print(f'    train_text_embs: {train_text_embs.shape}')

    # ── [2] Visium HD 임베딩 로드
    print('\n[2] Visium HD 임베딩 로드')
    val_img_embs = torch.tensor(np.load(hd_dir / 'tile_img_embs.npy')).float()
    val_emb_norm = F.normalize(val_img_embs, dim=-1)
    coords       = np.load(hd_dir / 'tile_coords.npy')
    print(f'    tile_img_embs: {val_img_embs.shape}')

    # ── [3] Visium HD 실제 발현값 로드
    print('\n[3] Visium HD 실제 발현값 로드')
    val_exprs = np.load(hd_dir / 'tile_exprs.npy')
    print(f'    tile_exprs: {val_exprs.shape}')

    # ── [4] Median filtering (SEQUOIA 방식)
    print('\n[4] Median filtering (3×3) 적용')
    val_exprs = apply_spatial_median_filter(val_exprs, coords)
    print(f'    완료')

    # ── [5] Unique expression values 필터 (SEQUOIA 방식)
    print('\n[5] Unique expression values 필터 (최소 10개)')
    unique_counts = np.array([len(np.unique(val_exprs[:, g])) for g in range(val_exprs.shape[1])])
    valid_genes   = unique_counts >= args.min_unique
    print(f'    유효 유전자: {valid_genes.sum()} / {len(valid_genes)}')

    # ── [6] Top 300 expressed genes 선택
    mean_expr  = val_exprs.mean(axis=0)
    top300_idx = np.argsort(mean_expr)[::-1][:300]
    # unique filter 적용
    top300_idx = np.array([i for i in top300_idx if valid_genes[i]])
    print(f'\n★ Top 300 genes (unique filter 후): {len(top300_idx)}개')

    # ── [7] 예측
    print(f'\n[6] 예측 시작 (pred_style={args.pred_style}, top_k={args.top_k})')
    n           = len(val_emb_norm)
    predictions = np.zeros((n, train_exprs.shape[1]), dtype=np.float32)

    for start in tqdm(range(0, n, args.batch_size), desc='Predicting'):
        end = min(start + args.batch_size, n)
        predictions[start:end] = predict_batch(
            val_emb_norm[start:end], train_emb_norm, train_exprs,
            args.pred_style, args.temperature, args.top_k, device
        )
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # ── [8] Top 300 슬라이싱
    predictions_top = predictions[:, top300_idx]
    val_exprs_top   = val_exprs[:, top300_idx]

    # ── [9] Percentile 정규화 (SEQUOIA 방식)
    print('\n[7] Percentile 정규화 (0~100) 적용')
    predictions_top = to_percentile(predictions_top)
    val_exprs_top   = to_percentile(val_exprs_top)

    # ── [10] PCC 계산
    print('\n[8] PCC 계산')
    spot_corrs = calc_spot_pcc(predictions_top, val_exprs_top)
    gene_corrs = calc_gene_pcc(predictions_top, val_exprs_top)

    print('\n' + '='*60)
    print(f'pred_style:  {args.pred_style}')
    print(f'top_k:       {args.top_k}')
    print(f'min_unique:  {args.min_unique}')
    print(f'eval genes:  {len(top300_idx)}개 (top 300 + unique filter)')
    print('='*60)
    print(f'Spot-wise PCC: mean={spot_corrs.mean():.4f}, median={np.median(spot_corrs):.4f}')
    print(f'Gene-wise PCC: mean={gene_corrs.mean():.4f}, median={np.median(gene_corrs):.4f}')
    print('='*60)

    # ── [11] 저장
    out_path = hd_dir / 'visium_hd_pcc_results_v2.npy'
    np.save(out_path, {'spot_corrs': spot_corrs, 'gene_corrs': gene_corrs})
    print(f'\n✅ 결과 저장: {out_path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--train_emb_dir', required=True)
    p.add_argument('--visium_hd_dir', required=True)
    p.add_argument('--pred_style',    default='loki', choices=['loki', 'softmax'])
    p.add_argument('--top_k',         type=int,   default=500)
    p.add_argument('--temperature',   type=float, default=0.07)
    p.add_argument('--batch_size',    type=int,   default=512)
    p.add_argument('--device',        type=str,   default='cuda:0')
    p.add_argument('--min_unique',    type=int,   default=10,
                   help='최소 unique expression values 수 (기본값=10)')
    args = p.parse_args()
    main(args)