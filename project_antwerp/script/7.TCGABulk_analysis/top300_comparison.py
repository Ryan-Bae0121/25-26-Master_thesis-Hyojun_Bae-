"""
Top 300 Gene 선택 전략 비교
============================

저장된 zero-shot 예측 결과(전체 1968 gene)를 재사용해서
세 가지 방식으로 top 300 gene을 선택하고 PCC를 비교한다.

방법 A: ST val_exprs 기준 top 300 (평균 발현량 높은 순) — 현재 방식
방법 B: TCGA bulk HVG top 300 (슬라이드 간 variance 높은 순)
방법 C: PCC top 300 (사후적 선택 — 상한선 참고용, 실제 적용 불가)
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import os

# ─── 경로 ────────────────────────────────────────────────────────────────────
ZEROSHOT_PRED = "/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/zeroshot"
ZEROSHOT_EMB  = "/project_antwerp/hbae/Loki_output/0228_embeddings_zeroshot/fold_01"
GENE_LIST     = "/project_antwerp/hbae/data/0228_HVG_NEW/0228_HVG_Finetune_gene_list_full.txt"
OUTPUT_DIR    = "/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/top300_comparison"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── 전체 예측값 로드 ─────────────────────────────────────────────────────────
print("Loading saved predictions (1968 genes)...")
pred_arr   = np.load(os.path.join(ZEROSHOT_PRED, "slide_preds_sliding_window.npy"))  # (331, 1968)
bulk_arr   = np.load(os.path.join(ZEROSHOT_PRED, "slide_bulks.npy"))                 # (331, 1968)
common_genes = np.load(os.path.join(ZEROSHOT_PRED, "common_genes.npy"), allow_pickle=True).tolist()
slide_ids  = np.load(os.path.join(ZEROSHOT_PRED, "slide_ids.npy"),    allow_pickle=True)

print(f"  pred_arr:  {pred_arr.shape}")
print(f"  bulk_arr:  {bulk_arr.shape}")
print(f"  genes:     {len(common_genes)}")
print(f"  slides:    {len(slide_ids)}")

# ─── Gene list 로드 ──────────────────────────────────────────────────────────
with open(GENE_LIST) as f:
    gene_list = [l.strip() for l in f if l.strip()]

# ─── PCC 계산 함수 ────────────────────────────────────────────────────────────
def compute_pcc(pred_arr, bulk_arr, gene_idx_list, gene_names, label):
    """
    gene_idx_list: common_genes에서의 index 리스트
    """
    p_sub = pred_arr[:, gene_idx_list]  # (S, 300)
    b_sub = bulk_arr[:, gene_idx_list]  # (S, 300)

    # Gene-wise PCC: gene X가 슬라이드 간 차이를 얼마나 잘 반영하는가
    gene_pccs, valid_genes = [], []
    for i in range(p_sub.shape[1]):
        p, b = p_sub[:, i], b_sub[:, i]
        if p.std() < 1e-8 or b.std() < 1e-8:
            continue
        r, _ = pearsonr(p, b)
        gene_pccs.append(r)
        valid_genes.append(gene_names[i])

    gene_pccs = np.array(gene_pccs)

    # Slide-wise PCC: 슬라이드 전체 gene profile이 얼마나 잘 맞는가
    slide_pccs = []
    for i in range(p_sub.shape[0]):
        p, b = p_sub[i], b_sub[i]
        if p.std() < 1e-8 or b.std() < 1e-8:
            slide_pccs.append(np.nan)
            continue
        r, _ = pearsonr(p, b)
        slide_pccs.append(r)

    slide_pccs = np.array(slide_pccs)
    valid = ~np.isnan(slide_pccs)

    print(f"\n[{label}]")
    print(f"  Genes used:  {len(valid_genes)}")
    print(f"  Gene-wise  | mean={gene_pccs.mean():.4f}  median={np.median(gene_pccs):.4f}  "
          f"PCC>0.1: {(gene_pccs>0.1).sum()}  PCC>0.3: {(gene_pccs>0.3).sum()}")
    print(f"  Slide-wise | mean={slide_pccs[valid].mean():.4f}  "
          f"median={np.median(slide_pccs[valid]):.4f}")

    return gene_pccs, slide_pccs, valid_genes


# ─── 방법 A: ST val_exprs 기준 top 300 ───────────────────────────────────────
print("\n" + "="*60)
print("Method A: ST val_exprs top 300 (현재 방식)")
print("="*60)

val_exprs    = np.load(os.path.join(ZEROSHOT_EMB, "val_exprs.npy"))  # (7215, 2000)
mean_expr    = val_exprs.mean(axis=0)                                  # (2000,)
top300_hvg_A = np.argsort(mean_expr)[::-1][:300]                      # HVG index
top300_genes_A = [gene_list[i] for i in top300_hvg_A]

# common_genes에서 방법 A gene의 index
idx_A = [i for i, g in enumerate(common_genes) if g in set(top300_genes_A)]
names_A = [common_genes[i] for i in idx_A]
print(f"  Top 300 genes in common_genes: {len(idx_A)}")
print(f"  Val expr range: {mean_expr[top300_hvg_A].min():.3f} ~ {mean_expr[top300_hvg_A].max():.3f}")

gene_pccs_A, slide_pccs_A, valid_genes_A = compute_pcc(
    pred_arr, bulk_arr, idx_A, names_A, "Method A: ST val_exprs top 300"
)


# ─── 방법 B: TCGA bulk HVG top 300 ───────────────────────────────────────────
print("\n" + "="*60)
print("Method B: TCGA bulk HVG top 300 (슬라이드 간 variance 기준)")
print("="*60)

bulk_var     = bulk_arr.var(axis=0)                    # (1968,) 슬라이드 간 variance
idx_B        = np.argsort(bulk_var)[::-1][:300].tolist()
names_B      = [common_genes[i] for i in idx_B]
print(f"  Bulk variance range: {bulk_var[idx_B].min():.4f} ~ {bulk_var[idx_B].max():.4f}")

gene_pccs_B, slide_pccs_B, valid_genes_B = compute_pcc(
    pred_arr, bulk_arr, idx_B, names_B, "Method B: TCGA bulk HVG top 300"
)


# ─── 방법 C: PCC top 300 (사후적 선택) ───────────────────────────────────────
print("\n" + "="*60)
print("Method C: PCC top 300 (post-hoc — 상한선 참고용)")
print("="*60)
print("  ⚠️  정답을 알고 gene을 선택하는 방식 → 실제 적용 불가")
print("  ⚠️  성능 상한선(upper bound) 확인 목적으로만 사용")

# 전체 1968 gene에 대해 gene-wise PCC 계산
all_gene_pccs = []
for i in range(len(common_genes)):
    p, b = pred_arr[:, i], bulk_arr[:, i]
    if p.std() < 1e-8 or b.std() < 1e-8:
        all_gene_pccs.append(-999)
        continue
    r, _ = pearsonr(p, b)
    all_gene_pccs.append(r)

all_gene_pccs = np.array(all_gene_pccs)

# PCC 높은 순으로 300개 선택
valid_mask   = all_gene_pccs > -999
valid_idx    = np.where(valid_mask)[0]
idx_C        = valid_idx[np.argsort(all_gene_pccs[valid_idx])[::-1][:300]].tolist()
names_C      = [common_genes[i] for i in idx_C]
print(f"  PCC range of selected genes: {all_gene_pccs[idx_C].min():.4f} ~ {all_gene_pccs[idx_C].max():.4f}")

gene_pccs_C, slide_pccs_C, valid_genes_C = compute_pcc(
    pred_arr, bulk_arr, idx_C, names_C, "Method C: PCC top 300 (post-hoc)"
)


# ─── 최종 비교 요약 ───────────────────────────────────────────────────────────
print("\n" + "="*60)
print("최종 비교 요약")
print("="*60)

valid_A = ~np.isnan(slide_pccs_A)
valid_B = ~np.isnan(slide_pccs_B)
valid_C = ~np.isnan(slide_pccs_C)

summary = pd.DataFrame({
    "방법": ["A: ST val_exprs top 300", "B: TCGA bulk HVG top 300", "C: PCC top 300 (post-hoc)"],
    "선택 기준": ["ST val 평균 발현량", "TCGA bulk variance", "실제 PCC 높은 순"],
    "실제 적용": ["가능", "가능", "불가 (상한선)"],
    "Gene-wise mean": [f"{gene_pccs_A.mean():.4f}", f"{gene_pccs_B.mean():.4f}", f"{gene_pccs_C.mean():.4f}"],
    "Gene-wise median": [f"{np.median(gene_pccs_A):.4f}", f"{np.median(gene_pccs_B):.4f}", f"{np.median(gene_pccs_C):.4f}"],
    "Slide-wise mean": [
        f"{slide_pccs_A[valid_A].mean():.4f}",
        f"{slide_pccs_B[valid_B].mean():.4f}",
        f"{slide_pccs_C[valid_C].mean():.4f}"
    ],
})

print(summary.to_string(index=False))

# ─── Gene 겹침 분석 ───────────────────────────────────────────────────────────
set_A = set(valid_genes_A)
set_B = set(valid_genes_B)
set_C = set(valid_genes_C)

print(f"\n[Gene 겹침 분석]")
print(f"  A ∩ B: {len(set_A & set_B)} genes")
print(f"  A ∩ C: {len(set_A & set_C)} genes")
print(f"  B ∩ C: {len(set_B & set_C)} genes")
print(f"  A ∩ B ∩ C: {len(set_A & set_B & set_C)} genes")

# ─── 저장 ─────────────────────────────────────────────────────────────────────
summary.to_csv(os.path.join(OUTPUT_DIR, "top300_comparison_summary.csv"), index=False)

for label, genes, pccs in [("A", valid_genes_A, gene_pccs_A),
                             ("B", valid_genes_B, gene_pccs_B),
                             ("C", valid_genes_C, gene_pccs_C)]:
    pd.DataFrame({"gene": genes, "pcc": pccs}).sort_values(
        "pcc", ascending=False
    ).to_csv(os.path.join(OUTPUT_DIR, f"gene_pcc_method_{label}.csv"), index=False)

print(f"\nSaved to {OUTPUT_DIR}")
print("Done!")