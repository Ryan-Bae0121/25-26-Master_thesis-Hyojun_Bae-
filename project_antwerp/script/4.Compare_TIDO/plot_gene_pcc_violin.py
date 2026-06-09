#!/usr/bin/env python3
"""
violin_pcc.py  ← 완전 새 파일명 (기존 plot_gene_pcc_violin.py 와 혼동 방지)
=============
sample_gene_pcc_dist.npy 를 읽어 샘플별 PCC violin plot 생성.
★ 각 샘플이 고유한 색상을 가짐 (tab20 palette)

Usage:
    python violin_pcc.py \
        --npy /project_antwerp/hbae/Loki_output/0228_10fold_finetune_embedding/fold_01/sample_gene_pcc_dist.npy \
        --output /tmp/fold01_violin.png

    # 여러 fold 비교 (패널별로 row 추가)
    python violin_pcc.py \
        --npy fold_01/.../sample_gene_pcc_dist.npy fold_06/.../sample_gene_pcc_dist.npy \
        --fold_labels "Fold 01" "Fold 06" \
        --output /tmp/compare.png
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ────────────────────────────────────────────────────────────
# 데이터셋 정렬 순서  (색상과 무관, 정렬에만 사용)
# ────────────────────────────────────────────────────────────
DATASET_ORDER = [
    'GSE181300', 'GSE208253', 'GSE220978',
    'GSE252265', 'GSE281978', 'Queensland', 'Zenodo',
]

# Queensland 샘플 ID 패턴
QUEENSLAND_PREFIXES = [
    'patient', 'visium_s', 'hn', 'hn0', 'hn1', 'hn2', 'hn3', 'hn4',
    'p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9',
]


def get_dataset(sid: str) -> str:
    """sample_id → dataset 이름 (정렬 전용)"""

    # ── GSM → GSE 직접 매핑 (실험에서 확인된 샘플들) ──────────
    # 새 샘플이 Unknown으로 뜨면 여기에 추가
    GSM_TO_GSE = {
        'GSM6339631_s1':               'GSE208253',
        'GSM6339631_S1':               'GSE208253',
        'GSM8633892_21_00758_LI_SING': 'GSE281978',
        'GSM8633892_21_00759_LI_SING': 'GSE281978',
    }
    if sid in GSM_TO_GSE:
        return GSM_TO_GSE[sid]

    sl = sid.lower()
    for ds in DATASET_ORDER:
        if ds.lower() in sl:
            return ds
    if any(sl.startswith(p) for p in QUEENSLAND_PREFIXES):
        return 'Queensland'
    if sid[0].isdigit():
        return 'Zenodo'
    return 'GEO_unknown'


def sort_samples(sids):
    def key(s):
        ds = get_dataset(s)
        rank = DATASET_ORDER.index(ds) if ds in DATASET_ORDER else len(DATASET_ORDER)
        return (rank, s)
    return sorted(sids, key=key)


# ────────────────────────────────────────────────────────────
# 샘플별 색상 팔레트 (tab20 × 3 = 최대 60색)
# ────────────────────────────────────────────────────────────
def make_palette(sids):
    """각 sample_id 에 고유 색상 할당 → dict 반환"""
    palette = []
    for cmap_name in ['tab20', 'tab20b', 'tab20c']:
        cm = plt.get_cmap(cmap_name)
        palette += [cm(i) for i in range(cm.N)]
    return {sid: palette[i % len(palette)] for i, sid in enumerate(sids)}


# ────────────────────────────────────────────────────────────
# 단일 axes 에 violin 그리기
# ────────────────────────────────────────────────────────────
def draw_panel(ax, sids, data, label, pcc_type='gene'):
    n = len(sids)
    positions   = list(range(1, n + 1))
    vdata       = [data[s] for s in sids]
    palette     = make_palette(sids)          # ← 샘플별 색상
    colors      = [palette[s] for s in sids]

    # ── violin body ──────────────────────────────────────────
    vp = ax.violinplot(vdata, positions=positions,
                       showmeans=True, showmedians=True,
                       showextrema=True, widths=0.72)

    for body, c in zip(vp['bodies'], colors):
        body.set_facecolor(c)
        body.set_alpha(0.75)
        body.set_edgecolor('black')
        body.set_linewidth(0.5)

    for key in ('cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxes'):
        if key in vp:
            vp[key].set_color('black')
            vp[key].set_linewidth(1.5 if key == 'cmeans' else 1.0)
            if key == 'cmeans':
                vp[key].set_linestyle('--')

    # ── 데이터셋 경계 점선 ───────────────────────────────────
    prev_ds = None
    for i, s in enumerate(sids):
        ds = get_dataset(s)
        if prev_ds is not None and ds != prev_ds:
            ax.axvline(i + 0.5, color='#999', linestyle='--',
                       linewidth=0.9, alpha=0.6)
        prev_ds = ds

    # ── 배경 홀짝 shading ────────────────────────────────────
    prev_ds, ds_start, tog = None, 1, False
    for i, s in enumerate(sids):
        ds = get_dataset(s)
        if prev_ds is not None and ds != prev_ds:
            if tog:
                ax.axvspan(ds_start - 0.5, i + 0.5, color='k', alpha=0.04)
            tog = not tog
            ds_start = i + 1
        prev_ds = ds
    if tog:
        ax.axvspan(ds_start - 0.5, n + 0.5, color='k', alpha=0.04)

    # ── y=0 기준선 ───────────────────────────────────────────
    ax.axhline(0, color='red', linestyle=':', linewidth=0.9, alpha=0.6)
    ax.grid(axis='y', linestyle='--', alpha=0.35)

    # ── X축 레이블 ───────────────────────────────────────────
    ax.set_xticks(positions)
    ax.set_xticklabels(sids, rotation=55, ha='right', fontsize=7)
    ax.set_xlim(0.2, n + 0.8)

    # ── Y축 ─────────────────────────────────────────────────
    ylabel = 'Gene-wise PCC' if pcc_type == 'gene' else 'Spot-wise PCC'
    ax.set_ylabel(ylabel, fontsize=10)

    # ── Title ────────────────────────────────────────────────
    ptype_str = 'Gene' if pcc_type == 'gene' else 'Spot'
    ax.set_title(f"{label}  ·  {ptype_str}-wise PCC per Sample (Top 300 genes)",
                 fontsize=11, fontweight='bold', pad=6)

    # ── 범례: 샘플별 고유 색 ─────────────────────────────────
    handles = [
        mpatches.Patch(facecolor=palette[s], label=s,
                       alpha=0.85, edgecolor='black', linewidth=0.4)
        for s in sids
    ]
    ncol = max(1, n // 20)
    ax.legend(handles=handles, loc='upper right', fontsize=7,
              framealpha=0.88, edgecolor='#ccc',
              ncol=ncol, handlelength=1.2, labelspacing=0.35)


# ────────────────────────────────────────────────────────────
# main
# ────────────────────────────────────────────────────────────
def main(args):
    npy_paths   = args.npy
    fold_labels = args.fold_labels or [Path(p).parent.name for p in npy_paths]
    assert len(npy_paths) == len(fold_labels)

    # 데이터 로드
    all_data, all_sids = [], []
    for path in npy_paths:
        print(f"Loading: {path}")
        d    = np.load(path, allow_pickle=True).item()
        sids = sort_samples(list(d.keys()))
        all_data.append(d)
        all_sids.append(sids)
        print(f"  Samples ({len(sids)}): {sids}")
        for s in sids:
            arr = d[s]
            print(f"    {s:40s}  n={len(arr):3d}  "
                  f"mean={arr.mean():.4f}  median={np.median(arr):.4f}")

    # figure 크기
    n_panels    = len(npy_paths)
    max_samples = max(len(s) for s in all_sids)
    fw          = max(14, max_samples * 0.6)
    fh          = 5.2 * n_panels + 0.5 * (n_panels - 1)

    fig, axes = plt.subplots(n_panels, 1, figsize=(fw, fh),
                             squeeze=False)
    axes = axes.flatten()

    # 공통 ylim (--share_ylim 시)
    if args.share_ylim and n_panels > 1:
        all_vals  = [v for d in all_data for arr in d.values() for v in arr]
        lo, hi    = np.percentile(all_vals, 1), np.percentile(all_vals, 99)
        pad       = (hi - lo) * 0.08
        ylim      = (lo - pad, hi + pad)
    else:
        ylim = None

    for ax, data, sids, label in zip(axes, all_data, all_sids, fold_labels):
        draw_panel(ax, sids, data, label, pcc_type=args.pcc_type)
        if ylim:
            ax.set_ylim(*ylim)

    plt.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved: {out}")
    plt.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--npy',        nargs='+', required=True)
    ap.add_argument('--fold_labels',nargs='+', default=None)
    ap.add_argument('--pcc_type',   default='gene', choices=['gene', 'spot'])
    ap.add_argument('--output',     default='violin_pcc.png')
    ap.add_argument('--share_ylim', action='store_true')
    main(ap.parse_args())