#!/usr/bin/env python3
"""
predict_per_sample.py
=====================
predict_fast.py와 동일한 수식으로 예측하되,
val CSV를 이용해 샘플별로 Gene-wise PCC를 분리 출력
★ 평가는 validation set 기준 top 300 expressed genes만 사용 (Loki 논문 방식)

Usage:
    python predict_per_sample.py \
        --emb_dir /project_antwerp/hbae/Loki_output/0228_10fold_finetune_embedding/fold_03 \
        --val_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/0228_New_HVG_10fold/fold_03_val.csv \
        --pred_style loki \
        --top_k 50
    python predict_per_sample.py \
        --emb_dir /project_antwerp/hbae/Loki_output/0228_epoch10_finetune_embedding/fold_01 \
        --val_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/0228_New_HVG_10fold/fold_01_val.csv \
        --pred_style loki \
        --top_k 500

Usage:

nohup python predict_per_sample.py \
        --emb_dir /project_antwerp/hbae/Loki_output/0228_10fold_finetune_embedding/fold_01 \
        --val_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/0228_New_HVG_10fold/fold_01_val.csv \
        --pred_style loki \
        --top_k 500
        --device cuda:1
        > /tmp/fin_fold01_per_sample.txt 2>&1 &

        
        python predict_per_sample.py \
        --emb_dir /project_antwerp/hbae/Loki_output/0228_10fold_finetune_embedding/fold_06 \
        --val_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/0228_New_HVG_10fold/fold_06_val.csv \
        --pred_style loki
        --top_k 500
        
        python predict_per_sample.py \
        --emb_dir /project_antwerp/hbae/Loki_output/0228_10fold_finetune_embedding/fold_02 \
        --val_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/0228_New_HVG_10fold/fold_02_val.csv \
        --pred_style loki
        --top_k 500
        
        python predict_per_sample.py \
        --emb_dir /project_antwerp/hbae/Loki_output/0228_10fold_finetune_embedding/fold_01 \
        --val_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/0228_New_HVG_10fold/fold_01_val.csv \
        --pred_style loki

New 버전
        python predict_per_sample.py \
        --emb_dir /project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new/fold_01 \
        --val_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/0317_New_HVG_10fold/fold_01_val.csv \
        --pred_style loki

        python predict_per_sample.py \
        --emb_dir /project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new/fold_02 \
        --val_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/0317_New_HVG_10fold/fold_02_val.csv \
        --pred_style loki

        python predict_per_sample.py \
        --emb_dir /project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new/fold_03 \
        --val_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/0317_New_HVG_10fold/fold_03_val.csv \
        --pred_style loki

python predict_per_sample.py \
        --emb_dir /project_antwerp/hbae/Loki_output/65um_finetune_embedding/fold_03 \
        --val_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/65um_fold_03/fold_03_val_65um.csv \
        --pred_style loki

norm 버전
        python predict_per_sample.py \
        --emb_dir /project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_norm/fold_01 \
        --val_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/0317_New_HVG_10fold/fold_01_val.csv \
        --pred_style loki
# 3. 진행 확인
tail -f /tmp/fin_fold01_per_sample.txt

"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from tqdm import tqdm
import pandas as pd


def predict(val_emb, train_emb, train_exprs, pred_style, temperature, top_k, return_debug=False):
    sim = val_emb @ train_emb.T  # (N,)

    if top_k is not None:
        idx = sim.topk(top_k).indices
        sim = sim[idx]
        train_exprs = train_exprs[idx]

    # ---- weight 계산 ----
    if pred_style in ("loki", "case_study", "img2img"):
        # linear normalization (Eq 16 스타일)
        s = sim.sum()
        w = sim / s if s.abs() > 1e-12 else torch.ones_like(sim) / sim.numel()

    elif pred_style in ("softmax", "img2img_softmax"):
        # softmax weight
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


def extract_sample_id(img_path: str) -> str:
    """
    경로에서 sample_id 추출.
    구조: .../Processed_Data/{dataset}/{sample_id}/patches/xxx.png
    GSM prefix 없는 샘플(Patient1, Visium_S01, P5, 17B5776 등)도 정상 처리.
    """
    parts = Path(img_path).parts
    if 'Processed_Data' in parts:
        idx = parts.index('Processed_Data')
        return parts[idx + 2]  # dataset 다음이 sample_id
    # fallback: patches 폴더의 부모
    return Path(img_path).parent.parent.name

def summarize_weights(sim: torch.Tensor, w: torch.Tensor):
    """
    sim: (N,) cosine or dot
    w:   (N,) weights after normalization (loki or softmax)
    """
    sim_min = sim.min().item()
    sim_max = sim.max().item()
    sim_mean = sim.mean().item()
    neg_ratio = (sim < 0).float().mean().item()
    sim_sum = sim.sum().item()

    # Effective sample size (ESS): 1 / sum(w^2)
    # softmax에서는 항상 유효, loki는 음수 weight 때문에 해석이 애매할 수 있어도 "집중도" 감은 줌
    w2 = (w * w).sum().item()
    ess = (1.0 / w2) if w2 > 1e-12 else float("inf")

    # top-1 weight / top-5 weight mass (집중도)
    absw = w.abs()
    top1 = absw.max().item()
    top5 = absw.topk(min(5, absw.numel())).values.sum().item()

    return {
        "sim_min": sim_min,
        "sim_max": sim_max,
        "sim_mean": sim_mean,
        "neg_ratio": neg_ratio,
        "sim_sum": sim_sum,
        "ess": ess,
        "absw_top1": top1,
        "absw_top5_mass": top5,
        "N": int(sim.numel()),
    }


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
    print(f"  val_img_embs:    {val_img_embs.shape}")
    print(f"  val_exprs:       {val_exprs.shape}")

    print("\n[Similarity distribution check - first val sample]")

    # 반드시 cosine이 되도록 normalize (안전)
    train_emb_norm = F.normalize(train_text_embs, dim=-1)
    val_emb_norm   = F.normalize(val_img_embs, dim=-1)
    
    sim = val_emb_norm[0] @ train_emb_norm.T  # (N_train,)
    
    print("  min:", sim.min().item())
    print("  max:", sim.max().item())
    print("  mean:", sim.mean().item())
    print("  neg_ratio:", (sim < 0).float().mean().item())
    print("  sum:", sim.sum().item())
    print("-" * 50)

    # Load val CSV → sample_id 매핑 (GSM + non-GSM 모두 지원)
    print(f"\nLoading val CSV: {args.val_csv}")
    df = pd.read_csv(args.val_csv)
    df['sample_id'] = df['img_path'].apply(extract_sample_id)
    df = df.reset_index(drop=True)

    assert len(df) == len(val_img_embs), \
        f"CSV rows ({len(df)}) != val embeddings ({len(val_img_embs)})"

    # 샘플별 구성 출력
    print("\nVal sample breakdown:")
    for sid, group in df.groupby('sample_id'):
        print(f"  {sid}: {len(group):,} tiles")

    # ★ Top 300 expressed genes (validation set 기준, Loki 논문 방식)
    mean_expr = val_exprs.mean(axis=0)          # (n_genes,)
    top300_idx = np.argsort(mean_expr)[::-1][:300]
    print(f"\n★ Top 300 genes selected from val set (out of {val_exprs.shape[1]} HVG genes)")

    # train embedding 선택
    if args.pred_style in ("img2img", "img2img_softmax"):
        train_emb = train_img_embs
        print("[Mode] val IMAGE → train IMAGE similarity")
    else:
        train_emb = train_text_embs
        print(f"[Mode] val IMAGE → train TEXT similarity ({args.pred_style})")
    # -------------------------
    # Similarity/weight diagnostics per sample (quick)
    # -------------------------
    print("\n[Diagnostics] Similarity/weight stats per sample (first 3 tiles each)")

    # cosine 보장(저장된 emb가 이미 normalize돼 있어도 한번 더 안전하게)
    train_emb_diag = F.normalize(train_emb, dim=-1)
    val_emb_diag   = F.normalize(val_img_embs, dim=-1)

    for sid, group in df.groupby('sample_id'):
        idxs = group.index.tolist()[:3]  # 첫 3개만
        print(f"\n  Sample: {sid} (showing {len(idxs)} tiles)")
        for k, i in enumerate(idxs):
            _, sim_dbg, w_dbg = predict(
                val_emb_diag[i], train_emb_diag, train_exprs,
                pred_style=args.pred_style,
                temperature=args.temperature,
                top_k=args.top_k,
                return_debug=True
            )
            stats = summarize_weights(sim_dbg, w_dbg)
            print(
                f"    tile#{k+1} idx={i} | "
                f"sim[min,max,mean,sum]=[{stats['sim_min']:.3f},{stats['sim_max']:.3f},{stats['sim_mean']:.3f},{stats['sim_sum']:.1f}] "
                f"neg={stats['neg_ratio']:.3f} | "
                f"ESS~{stats['ess']:.1f} | "
                f"|w| top1={stats['absw_top1']:.4f}, top5_mass={stats['absw_top5_mass']:.4f}"
            )

    print("\n" + "-" * 50)

    # -------------------------
    # Predict (전체)
    # -------------------------
    # 예측에서도 cosine이 되도록 normalize 적용(논문 정의와 정합)
    train_emb_pred = F.normalize(train_emb, dim=-1)
    val_emb_pred   = F.normalize(val_img_embs, dim=-1)

    predictions = []
    for i in tqdm(range(len(val_img_embs)), desc="Predicting"):
        pred = predict(
            val_emb_pred[i], train_emb_pred, train_exprs,
            pred_style=args.pred_style,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        predictions.append(pred.cpu().numpy())
    predictions = np.array(predictions)  # (n_spots, n_genes)

    # ★ Top 300만 슬라이싱
    predictions_top300 = predictions[:, top300_idx]
    val_exprs_top300   = val_exprs[:, top300_idx]

    # 전체 평가 (top 300)
    spot_corrs = calc_spot_pcc(predictions_top300, val_exprs_top300)
    gene_corrs = calc_gene_pcc(predictions_top300, val_exprs_top300)

    print("\n" + "="*60)
    print(f"pred_style:  {args.pred_style}")
    print(f"top_k:       {args.top_k}")
    print(f"temperature: {args.temperature}")
    print(f"eval genes:  top 300 expressed in val set")
    print("="*60)
    print(f"[전체] Spot-wise PCC: mean={spot_corrs.mean():.4f}, median={np.median(spot_corrs):.4f}")
    print(f"[전체] Gene-wise PCC: mean={gene_corrs.mean():.4f}, median={np.median(gene_corrs):.4f}")

    # 샘플별 평가 (top 300)
   # print("\n" + "-"*60)기존꺼
    #print("샘플별 Gene-wise PCC (top 300 genes):")
    #print("-"*60)
    #for sid, group in df.groupby('sample_id'):
     #   idx = group.index.tolist()
      #  p = predictions_top300[idx]
       # v = val_exprs_top300[idx]
        #g_corrs = calc_gene_pcc(p, v)
        #s_corrs = calc_spot_pcc(p, v)
        #n = len(idx)
        #print(f"  {sid} (n={n:,})")
        #print(f"    Spot-wise PCC: mean={s_corrs.mean():.4f}")
        #print(f"    Gene-wise PCC: mean={g_corrs.mean():.4f}  (genes evaluated: {len(g_corrs)})")
    #print("="*60)
    # 샘플별 평가 (top 300) + gene PCC 분포 저장
    print("\n" + "-"*60)
    print("샘플별 Gene-wise PCC (top 300 genes):")
    print("-"*60)
    
    sample_gene_pccs = {}  # ← 이걸 추가
    
    for sid, group in df.groupby('sample_id'):
        idx = group.index.tolist()
        p = predictions_top300[idx]
        v = val_exprs_top300[idx]
        g_corrs = calc_gene_pcc(p, v)
        s_corrs = calc_spot_pcc(p, v)
        n = len(idx)
        sample_gene_pccs[sid] = g_corrs  # ← 이걸 추가
        print(f"  {sid} (n={n:,})")
        print(f"    Spot-wise PCC: mean={s_corrs.mean():.4f}")
        print(f"    Gene-wise PCC: mean={g_corrs.mean():.4f}  (genes evaluated: {len(g_corrs)})")

    # 저장 ← 이 블록 추가
    out_path = Path(args.emb_dir) / 'Top_500_sample_gene_pcc_dist.npy'
    np.save(out_path, sample_gene_pccs)
    print(f"\n✅ Saved gene PCC distributions to: {out_path}")
    print("="*60)



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--emb_dir",     required=True)
    p.add_argument("--val_csv",     required=True)
    p.add_argument("--pred_style",  default="loki",
                   choices=["loki", "case_study", "softmax", "img2img","img2img_softmax"])
    p.add_argument("--top_k",       type=int, default=None)
    p.add_argument("--temperature", type=float, default=0.07)
    args = p.parse_args()
    main(args)