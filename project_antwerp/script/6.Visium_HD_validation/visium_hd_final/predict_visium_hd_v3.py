#!/usr/bin/env python3
"""
predict_visium_hd_v3.py
========================
HVG 2000개 전체로 평가
- top 300 제한 없음
- Visium HD tile_exprs의 모든 HVG 유전자로 평가
- (HVG에 없는 유전자는 자동으로 skip됨)

Usage:
    python predict_visium_hd_v3.py \
        --train_emb_dir /project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new/fold_01 \
        --visium_hd_dir /project_antwerp/hbae/Loki_output/visium_hd_array_embeddings/fold_01 \
        --pred_style loki \
        --device cuda:0

        
        python predict_visium_hd_v3.py \
    --train_emb_dir /project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new/fold_01 \
    --visium_hd_dir /project_antwerp/hbae/Loki_output/visium_hd_array_embeddings/fold_01 \
    --pred_style loki \
    --top_k 500 \
    --batch_size 512 \
    --device cuda:0

    (base) hyobaeug@715b2cb18487:/project_antwerp/hbae/script/0208_start/Visium_HD$ python predict_visium_hd_v3.py \
        --train_emb_dir /project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new/fold_01 \
        --visium_hd_dir /project_antwerp/hbae/Loki_output/visium_hd_array_embeddings/fold_01 \
        --pred_style loki \
        --device cuda:0

[1] train 임베딩 로드
    train_emb:   torch.Size([49156, 768])
    train_exprs: torch.Size([49156, 2000])

[2] Visium HD 임베딩 로드
    tile_img_embs: torch.Size([8649, 768])

[3] Visium HD 발현값 로드
    tile_exprs: (8649, 2000)
    zeros 비율: 0.4903

★ 평가 유전자: HVG 2000개 전체 (top 300 제한 없음)
   (발현값이 uniform한 유전자는 std=0 → 자동 skip)

[4] 예측 (pred_style=loki, top_k=None)
Predicting: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 17/17 [00:10<00:00,  1.61it/s]

[5] Variation 확인
    예측값 std (tile간): 0.0085
    실제값 std (tile간): 0.6114

[6] PCC 계산

============================================================
pred_style:  loki
top_k:       None
eval 유전자: HVG 2000개 전체
  (std=0 유전자 skip → 실제 평가: 2000개)
============================================================
Spot-wise PCC: mean=0.2781, median=0.2392
Gene-wise PCC: mean=0.0085, median=0.0070
============================================================

Gene-wise PCC 상위 10개:
  LAMB3           PCC=0.1082  mean_expr=0.684
  CD8A            PCC=0.1011  mean_expr=2.012
  SELENOP         PCC=0.1010  mean_expr=1.778
  CD6             PCC=0.0979  mean_expr=2.516
  MSLN            PCC=0.0973  mean_expr=0.572
  ZAP70           PCC=0.0962  mean_expr=2.491
  GIMAP7          PCC=0.0956  mean_expr=2.085
  PTGDS           PCC=0.0938  mean_expr=3.612
  KLRK1           PCC=0.0931  mean_expr=0.920
  TRAF3IP3        PCC=0.0897  mean_expr=2.672

✅ 결과 저장: /project_antwerp/hbae/Loki_output/visium_hd_array_embeddings/fold_01/visium_hd_pcc_all2000_loki.npy
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
    print('\n[1] train 임베딩 로드')
    train_exprs = torch.tensor(np.load(train_dir / 'train_exprs.npy')).float()
    if args.pred_style == 'img2img':
        train_emb = torch.tensor(np.load(train_dir / 'train_img_embs.npy')).float()
    else:
        train_emb = torch.tensor(np.load(train_dir / 'train_text_embs.npy')).float()
    print(f'    train_emb:   {train_emb.shape}')
    print(f'    train_exprs: {train_exprs.shape}')
    train_emb_norm = F.normalize(train_emb, dim=-1)

    # ── [2] Visium HD 임베딩 + 발현값 로드
    print('\n[2] Visium HD 임베딩 로드')
    val_img_embs = torch.tensor(np.load(hd_dir / 'tile_img_embs.npy')).float()
    val_emb_norm = F.normalize(val_img_embs, dim=-1)
    print(f'    tile_img_embs: {val_img_embs.shape}')

    print('\n[3] Visium HD 발현값 로드')
    val_exprs = np.load(hd_dir / 'tile_exprs.npy')
    print(f'    tile_exprs: {val_exprs.shape}')
    print(f'    zeros 비율: {(val_exprs==0).mean():.4f}')

    # ── [3] 평가 유전자: HVG 2000개 전체
    # tile_exprs가 이미 HVG 2000개 기준으로 저장됨
    # train_exprs도 HVG 2000개 기준
    # → 그냥 2000개 전체로 평가 (top 300 제한 없음)
    n_genes = val_exprs.shape[1]
    print(f'\n★ 평가 유전자: HVG {n_genes}개 전체 (top 300 제한 없음)')
    print(f'   (발현값이 uniform한 유전자는 std=0 → 자동 skip)')

    # ── [4] 예측
    print(f'\n[4] 예측 (pred_style={args.pred_style}, top_k={args.top_k})')
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

    # ── [5] Variation 확인
    print(f'\n[5] Variation 확인')
    print(f'    예측값 std (tile간): {predictions.std(axis=0).mean():.4f}')
    print(f'    실제값 std (tile간): {val_exprs.std(axis=0).mean():.4f}')

    # ── [6] PCC 계산 (2000개 전체, std=0인 유전자 skip)
    print('\n[6] PCC 계산')
    spot_corrs = calc_spot_pcc(predictions, val_exprs)
    gene_corrs = calc_gene_pcc(predictions, val_exprs)

    print('\n' + '='*60)
    print(f'pred_style:  {args.pred_style}')
    print(f'top_k:       {args.top_k}')
    print(f'eval 유전자: HVG 2000개 전체')
    print(f'  (std=0 유전자 skip → 실제 평가: {len(gene_corrs)}개)')
    print('='*60)
    print(f'Spot-wise PCC: mean={spot_corrs.mean():.4f}, median={np.median(spot_corrs):.4f}')
    print(f'Gene-wise PCC: mean={gene_corrs.mean():.4f}, median={np.median(gene_corrs):.4f}')
    print('='*60)

    # top 20 gene PCC
    gene_names = open('/project_antwerp/hbae/data/0317_hvg_2000_list.txt').read().strip().split('\n')
    gene_pcc_list = []
    for g in range(predictions.shape[1]):
        if val_exprs[:, g].std() > 1e-8:
            r, _ = pearsonr(predictions[:, g], val_exprs[:, g])
            if np.isfinite(r):
                gene_pcc_list.append((gene_names[g], r, val_exprs[:, g].mean()))
    gene_pcc_list.sort(key=lambda x: -x[1])
    print('\nGene-wise PCC 상위 10개:')
    for g, r, m in gene_pcc_list[:10]:
        print(f'  {g:<15} PCC={r:.4f}  mean_expr={m:.3f}')

    # ── [7] 저장
    out_path = hd_dir / f'visium_hd_pcc_all2000_{args.pred_style}.npy'
    np.save(out_path, {'spot_corrs': spot_corrs, 'gene_corrs': gene_corrs})
    print(f'\n✅ 결과 저장: {out_path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--train_emb_dir', required=True)
    p.add_argument('--visium_hd_dir', required=True)
    p.add_argument('--pred_style',    default='loki', choices=['loki', 'softmax', 'img2img'])
    p.add_argument('--top_k',         type=int,   default=None)
    p.add_argument('--temperature',   type=float, default=0.07)
    p.add_argument('--batch_size',    type=int,   default=512)
    p.add_argument('--device',        type=str,   default='cuda:0')
    args = p.parse_args()
    main(args)