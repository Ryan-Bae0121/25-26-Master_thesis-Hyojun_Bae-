#!/usr/bin/env python3
"""
predict_visium_hd.py
====================
Visium HD 유전자 발현 예측 + PCC 계산
pred_style: loki, softmax, img2img 지원

Usage:
    python predict_visium_hd.py \
        --train_emb_dir /project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new/fold_01 \
        --visium_hd_dir /project_antwerp/hbae/Loki_output/visium_hd_array_embeddings/fold_01 \
        --pred_style loki \
        --top_k 500 \
        --device cuda:0
python predict_visium_hd.py     --train_emb_dir /project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new/fold_01     --visium_hd_dir /project_antwerp/hbae/Loki_output/visium_hd_embeddings/fold_01     --pred_style loki     --batch_size 512     --device cuda:0

python predict_visium_hd.py     --train_emb_dir /project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new/fold_02     --visium_hd_dir /project_antwerp/hbae/Loki_output/visium_hd_embeddings/fold_02     --pred_style loki     --batch_size 512     --device cuda:0
python predict_visium_hd.py \
        --train_emb_dir /project_antwerp/hbae/Loki_output/0228_epoch10_finetune_embedding/fold_01 \
        --visium_hd_dir /project_antwerp/hbae/Loki_output/visium_hd_23px_2um_embeddings/fold_01 \
        --pred_style loki \
        --device cuda:0
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
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

        if pred_style in ('loki', 'img2img'):
            s = sim_k.sum()
            w = sim_k / s if s.abs() > 1e-12 else torch.ones_like(sim_k) / sim_k.numel()
        elif pred_style == 'softmax':
            w = F.softmax(sim_k / temperature, dim=0)

        preds.append((w[:, None] * expr_k).sum(dim=0).cpu())
    return torch.stack(preds).numpy()


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
    print('\n[1] train 임베딩 로드 (HNSCC fold)')
    train_exprs = torch.tensor(np.load(train_dir / 'train_exprs.npy')).float()

    if args.pred_style == 'img2img':
        train_emb = torch.tensor(np.load(train_dir / 'train_img_embs.npy')).float()
        print(f'    [img2img] train_img_embs: {train_emb.shape}')
    else:
        train_emb = torch.tensor(np.load(train_dir / 'train_text_embs.npy')).float()
        print(f'    train_text_embs: {train_emb.shape}')

    print(f'    train_exprs:     {train_exprs.shape}')
    train_emb_norm = F.normalize(train_emb, dim=-1)

    # ── [2] Visium HD 임베딩 로드
    print('\n[2] Visium HD 임베딩 로드')
    val_img_embs = torch.tensor(np.load(hd_dir / 'tile_img_embs.npy')).float()
    val_emb_norm = F.normalize(val_img_embs, dim=-1)
    print(f'    tile_img_embs: {val_img_embs.shape}')

    # ── [3] Visium HD 실제 발현값 로드
    print('\n[3] Visium HD 실제 발현값 로드')
    val_exprs = np.load(hd_dir / 'tile_exprs.npy')
    print(f'    tile_exprs: {val_exprs.shape}')
    print(f'    zeros 비율: {(val_exprs==0).mean():.4f}')

    # ── [4] top 300 expressed genes
    mean_expr  = val_exprs.mean(axis=0)
    top300_idx = np.argsort(mean_expr)[::-1][:300]
    print(f'\n★ Top 300 genes selected (out of {val_exprs.shape[1]} HVG genes)')

    # ── [5] 예측
    print(f'\n[4] 예측 시작 (pred_style={args.pred_style}, top_k={args.top_k})')
    print(f'    총 타일 수: {len(val_emb_norm):,}')

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

    # ── [6] top 300 슬라이싱
    predictions_top300 = predictions[:, top300_idx]
    val_exprs_top300   = val_exprs[:, top300_idx]

    # ── [7] 예측값 variation 확인
    print(f'\n[5] Variation 확인')
    print(f'    예측값 std (tile간): {predictions_top300.std(axis=0).mean():.4f}')
    print(f'    실제값 std (tile간): {val_exprs_top300.std(axis=0).mean():.4f}')

    # ── [8] PCC 계산
    print('\n[6] PCC 계산')
    spot_corrs = calc_spot_pcc(predictions_top300, val_exprs_top300)
    gene_corrs = calc_gene_pcc(predictions_top300, val_exprs_top300)

    print('\n' + '='*60)
    print(f'pred_style:  {args.pred_style}')
    print(f'top_k:       {args.top_k}')
    print(f'eval genes:  top 300 expressed in Visium HD')
    print('='*60)
    print(f'Spot-wise PCC: mean={spot_corrs.mean():.4f}, median={np.median(spot_corrs):.4f}')
    print(f'Gene-wise PCC: mean={gene_corrs.mean():.4f}, median={np.median(gene_corrs):.4f}')
    print('='*60)

    # ── [9] 저장
    out_path = hd_dir / f'visium_hd_pcc_{args.pred_style}.npy'
    np.save(out_path, {'spot_corrs': spot_corrs, 'gene_corrs': gene_corrs})
    print(f'\n✅ 결과 저장: {out_path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--train_emb_dir', required=True)
    p.add_argument('--visium_hd_dir', required=True)
    p.add_argument('--pred_style',    default='loki',
                   choices=['loki', 'softmax', 'img2img'])
    p.add_argument('--top_k',         type=int,   default=None)
    p.add_argument('--temperature',   type=float, default=0.07)
    p.add_argument('--batch_size',    type=int,   default=512)
    p.add_argument('--device',        type=str,   default='cuda:0')
    args = p.parse_args()
    main(args)