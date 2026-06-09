#!/usr/bin/env python3
"""
violin_pcc_merged.py
====================
fold_01 ~ fold_10 의 sample_gene_pcc_dist.npy 를 모두 읽어,
같은 샘플의 gene-wise PCC 값을 전 fold에 걸쳐 합산한 뒤
샘플별 violin plot 을 한 장에 그린다.

★ 각 violin = "해당 샘플이 val set 으로 등장한 모든 fold의 PCC 분포 합산"
★ 샘플별 고유 색상 (tab20 palette)
★ 데이터셋 순서: GSE181300 → GSE208253 → GSE220978 → GSE252265 → GSE281978 → Queensland → Zenodo

Usage:
    python violin_pcc_merged.py \
        --emb_dir /project_antwerp/hbae/Loki_output/0228_epoch10_finetune_embedding \
        --output  /tmp/all_folds_merged_violin.png

    # npy 파일명이 다르면 직접 지정
    python violin_pcc_merged.py \
        --emb_dir /project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new \
        --npy_name sample_gene_pcc_dist.npy \
        --output  /project_antwerp/hbae/script/0208_start/0317_remove_patient/all_folds_merged_violin.png
        
    python violin_pcc_merged.py \
        --emb_dir /project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new \
        --output  /project_antwerp/hbae/script/0208_start/0317_remove_patient/all_folds_merged_violin.png

    python violin_pcc_merged.py \
        --emb_dir /project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new \
        --output  /project_antwerp/hbae/script/0208_start/0317_K=30%/K=30%_all_folds_violin.png

    # spot PCC 버전
    python violin_pcc_merged.py \
        --emb_dir /project_antwerp/hbae/Loki_output/0228_epoch10_finetune_embedding \
        --npy_name sample_spot_pcc_dist.npy \
        --pcc_type spot \
        --output  /tmp/all_folds_spot_violin.png
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ────────────────────────────────────────────────────────────
# 데이터셋 정렬 순서
# ────────────────────────────────────────────────────────────
DATASET_ORDER = [
    'GSE181300', 'GSE208253', 'GSE220978',
    'GSE252265', 'GSE281978', 'Queensland', 'Zenodo',
]

QUEENSLAND_PREFIXES = [
    'visium_s', 'p5',
]

# ── GSM → GSE 직접 매핑 ────────────────────────────────────
GSM_TO_GSE = {
    # GSE181300
    'GSM5494475': 'GSE181300',
    'GSM5494476': 'GSE181300',
    'GSM5494477': 'GSE181300',
    'GSM5494478': 'GSE181300',
    'GSM5494479': 'GSE181300',
    'GSM5494480': 'GSE181300',
    'GSM5494481': 'GSE181300',
    'GSM5494482': 'GSE181300',
    # GSE208253
    'GSM6339631_s1':  'GSE208253',
    'GSM6339631_S1':  'GSE208253',
    'GSM6339632_s2':  'GSE208253',
    'GSM6339633_s3':  'GSE208253',
    'GSM6339634_s4':  'GSE208253',
    'GSM6339635_s5':  'GSE208253',
    'GSM6339636_s6':  'GSE208253',
    'GSM6339637_s7':  'GSE208253',
    'GSM6339638_s8':  'GSE208253',
    'GSM6339639_s9':  'GSE208253',
    'GSM6339640_s10': 'GSE208253',
    'GSM6339641_s11': 'GSE208253',
    'GSM6339642_s12': 'GSE208253',
    # GSE252265
    'GSM7998252': 'GSE252265',
    'GSM7998253': 'GSE252265',
    'GSM7998254': 'GSE252265',
    'GSM7998255': 'GSE252265',
    'GSM7998256': 'GSE252265',
    'GSM7998257': 'GSE252265',
    'GSM7998258': 'GSE252265',
    'GSM7998259': 'GSE252265',
    # GSE281978
    'GSM8633891_21_00757_LI_SING': 'GSE281978',
    'GSM8633892_21_00758_LI_SING': 'GSE281978',
    'GSM8633892_21_00759_LI_SING': 'GSE281978',
    'GSM8633893_21_01569_LI_SING': 'GSE281978',
    'GSM8633894_21_01570_LI_SING': 'GSE281978',
    'GSM8633895_21_01586_LI_SING': 'GSE281978',
    'GSM8633896_21_01587_LI_SING': 'GSE281978',
}


# ────────────────────────────────────────────────────────────
# Utilities
# ────────────────────────────────────────────────────────────
def get_dataset(sid: str) -> str:
    if sid in GSM_TO_GSE:
        return GSM_TO_GSE[sid]

    sl = sid.lower()

    # GSE220978: Patient1~4
    if sl in ('patient1', 'patient2', 'patient3', 'patient4'):
        return 'GSE220978'

    for ds in DATASET_ORDER:
        if ds.lower() in sl:
            return ds

    # Queensland: P5, Visium_S01
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


def make_palette(sids):
    """샘플별 고유 색상 dict"""
    palette = []
    for name in ['tab20', 'tab20b', 'tab20c']:
        cm = plt.get_cmap(name)
        palette += [cm(i) for i in range(cm.N)]
    return {sid: palette[i % len(palette)] for i, sid in enumerate(sids)}


# ────────────────────────────────────────────────────────────
# 데이터 로드 & 샘플별 합산
# ────────────────────────────────────────────────────────────
def load_and_merge(emb_dir: Path, npy_name: str) -> dict:
    """
    fold_01 ~ fold_10 의 npy 파일을 읽어
    { sample_id: np.array([모든 fold PCC 값 concat]) } 반환
    """
    merged = defaultdict(list)
    found_folds = []

    for fold_dir in sorted(emb_dir.glob('fold_*')):
        npy_path = fold_dir / npy_name
        if not npy_path.exists():
            print(f"  [skip] {npy_path} 없음")
            continue
        data = np.load(npy_path, allow_pickle=True).item()
        found_folds.append(fold_dir.name)
        for sid, arr in data.items():
            merged[sid].append(arr)

    print(f"\n✅ Loaded folds: {found_folds}")
    print(f"   Total unique samples: {len(merged)}\n")

    # 각 샘플의 PCC 배열을 fold 걸쳐 concat
    result = {}
    for sid, arrays in merged.items():
        result[sid] = np.concatenate(arrays)

    return result, found_folds


# ────────────────────────────────────────────────────────────
# Violin plot
# ────────────────────────────────────────────────────────────
def draw_violin(ax, sids, data, title, pcc_type, fold_count):
    n        = len(sids)
    pos      = list(range(1, n + 1))
    vdata    = [data[s] for s in sids]
    palette  = make_palette(sids)
    colors   = [palette[s] for s in sids]

    # ── violin body ──────────────────────────────────────────
    vp = ax.violinplot(vdata, positions=pos,
                       showmeans=True, showmedians=True,
                       showextrema=True, widths=0.75)

    for body, c in zip(vp['bodies'], colors):
        body.set_facecolor(c)
        body.set_alpha(0.75)
        body.set_edgecolor('black')
        body.set_linewidth(0.5)

    for k in ('cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxes'):
        if k in vp:
            vp[k].set_color('black')
            vp[k].set_linewidth(1.6 if k == 'cmeans' else 1.0)
            if k == 'cmeans':
                vp[k].set_linestyle('--')

    # ── 데이터셋 경계 점선 + 레이블 ─────────────────────────
    prev_ds = None
    for i, s in enumerate(sids):
        ds = get_dataset(s)
        if prev_ds is not None and ds != prev_ds:
            ax.axvline(i + 0.5, color='#888', linestyle='--',
                       linewidth=1.0, alpha=0.55)
            ax.text(i + 0.52, ax.get_ylim()[1] * 0.97,
                    ds, ha='left', va='top', fontsize=7,
                    color='#444', fontstyle='italic')
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

    # ── mean 값 + n_folds 텍스트 ─────────────────────────────
    ymax = ax.get_ylim()[1]
    for p, s in zip(pos, sids):
        mean_val = data[s].mean()
        ax.text(p, ymax, f"{mean_val:.3f}",
                ha='center', va='bottom', fontsize=6,
                color='#111', fontweight='bold')



    # ── 축 ───────────────────────────────────────────────────
    ax.set_xticks(pos)
    ax.set_xticklabels(sids, rotation=55, ha='right', fontsize=7.5)
    ax.set_xlim(0.2, n + 0.8)
    ylabel = 'Gene-wise PCC' if pcc_type == 'gene' else 'Spot-wise PCC'
    ax.set_ylabel(ylabel, fontsize=11)
    ax.axhline(0, color='red', linestyle=':', linewidth=0.9, alpha=0.6)
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=8)

    # ── 범례: 샘플별 ─────────────────────────────────────────
    handles = [
        mpatches.Patch(facecolor=palette[s], label=s,
                       alpha=0.85, edgecolor='black', linewidth=0.4)
        for s in sids
    ]
    ax.legend(handles=handles,
              loc='upper left', bbox_to_anchor=(1.01, 1.0),
              fontsize=7.5, framealpha=0.9, edgecolor='#ccc',
              ncol=1, handlelength=1.2, labelspacing=0.35,
              borderpad=0.7)


# ────────────────────────────────────────────────────────────
# main
# ────────────────────────────────────────────────────────────
def main(args):
    emb_dir = Path(args.emb_dir)
    assert emb_dir.exists(), f"디렉터리 없음: {emb_dir}"

    # 데이터 로드 & 합산
    merged_data, found_folds = load_and_merge(emb_dir, args.npy_name)

    if not merged_data:
        print("❌ 로드된 데이터 없음. --emb_dir 와 --npy_name 을 확인하세요.")
        return

    sids = sort_samples(list(merged_data.keys()))

    # 콘솔 요약
    ptype = args.pcc_type
    print(f"{'Sample':<40} {'Dataset':<12} {'n_vals':>7} "
          f"{'mean':>8} {'median':>8} {'std':>8}")
    print("-" * 85)
    for s in sids:
        arr = merged_data[s]
        print(f"{s:<40} {get_dataset(s):<12} {len(arr):>7} "
              f"{arr.mean():>8.4f} {np.median(arr):>8.4f} {arr.std():>8.4f}")

    # figure
    n          = len(sids)
    fw         = max(16, n * 0.65)
    fh         = 6.5
    fig, ax    = plt.subplots(figsize=(fw, fh))

    n_folds    = len(found_folds)
    ptype_str  = 'Gene' if ptype == 'gene' else 'Spot'
    title      = (f"{ptype_str}-wise PCC per Sample  "
                  f"(Top 300 genes · {n_folds} folds merged · {n} samples)")

    draw_violin(ax, sids, merged_data, title, ptype, n_folds)

    plt.tight_layout()
    plt.subplots_adjust(right=0.78)  # 오른쪽 범례 공간 확보
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved: {out}")
    plt.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description="10-fold PCC 분포를 샘플별로 합산한 violin plot"
    )
    ap.add_argument('--emb_dir',  required=True,
                    help='fold_01 ~ fold_10 디렉터리가 들어있는 상위 경로')
    ap.add_argument('--npy_name', default='Top_500_sample_gene_pcc_dist.npy',
                    help='각 fold 디렉터리 안의 npy 파일명 (default: sample_gene_pcc_dist.npy)')
    ap.add_argument('--pcc_type', default='gene', choices=['gene', 'spot'])
    ap.add_argument('--output',   default='merged_violin.png')
    main(ap.parse_args())