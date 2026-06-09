#!/usr/bin/env python3
"""
violin_compare.py
=================
내 모델 (violin_pcc_merged) 결과와 Tido 모델 결과를
위아래 2-panel로 나란히 비교하는 violin plot.

Usage:
    python violin_compare.py \
        --my_emb_dir /project_antwerp/hbae/Loki_output/0228_epoch10_finetune_embedding \
        --my_npy_name Top_500_sample_gene_pcc_dist.npy \
        --tido_npy /tmp/tido_sample_gene_pcc_dist.npy \
        --output /tmp/compare_violin.png
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
# Dataset 정의
# ────────────────────────────────────────────────────────────
DATASET_ORDER = [
    'GSE181300', 'GSE208253', 'GSE220978',
    'GSE252265', 'GSE281978', 'Queensland', 'Zenodo',
]

GSM_TO_GSE = {
    'GSM5494475': 'GSE181300', 'GSM5494476': 'GSE181300',
    'GSM5494477': 'GSE181300', 'GSM5494478': 'GSE181300',
    'GSM5494479': 'GSE181300', 'GSM5494480': 'GSE181300',
    'GSM5494481': 'GSE181300', 'GSM5494482': 'GSE181300',
    'GSM6339631_s1':  'GSE208253', 'GSM6339631_S1':  'GSE208253',
    'GSM6339632_s2':  'GSE208253', 'GSM6339633_s3':  'GSE208253',
    'GSM6339634_s4':  'GSE208253', 'GSM6339635_s5':  'GSE208253',
    'GSM6339636_s6':  'GSE208253', 'GSM6339637_s7':  'GSE208253',
    'GSM6339638_s8':  'GSE208253', 'GSM6339639_s9':  'GSE208253',
    'GSM6339640_s10': 'GSE208253', 'GSM6339641_s11': 'GSE208253',
    'GSM6339642_s12': 'GSE208253',
    'GSM7998252': 'GSE252265', 'GSM7998253': 'GSE252265',
    'GSM7998254': 'GSE252265', 'GSM7998255': 'GSE252265',
    'GSM7998256': 'GSE252265', 'GSM7998257': 'GSE252265',
    'GSM7998258': 'GSE252265', 'GSM7998259': 'GSE252265',
    'GSM8633891_21_00757_LI_SING': 'GSE281978',
    'GSM8633892_21_00758_LI_SING': 'GSE281978',
    'GSM8633892_21_00759_LI_SING': 'GSE281978',
    'GSM8633893_21_01569_LI_SING': 'GSE281978',
    'GSM8633894_21_01570_LI_SING': 'GSE281978',
    'GSM8633895_21_01586_LI_SING': 'GSE281978',
    'GSM8633896_21_01587_LI_SING': 'GSE281978',
}

QUEENSLAND_PREFIXES = ['visium_s', 'p5']


def get_dataset(sid: str) -> str:
    if sid in GSM_TO_GSE:
        return GSM_TO_GSE[sid]
    sl = sid.lower()
    if sl in ('patient1', 'patient2', 'patient3', 'patient4'):
        return 'GSE220978'
    for ds in DATASET_ORDER:
        if ds.lower() in sl:
            return ds
    if any(sl.startswith(p) for p in QUEENSLAND_PREFIXES):
        return 'Queensland'
    if sid[0].isdigit():
        return 'Zenodo'
    return 'GEO_unknown'


def normalize_sid(sid: str) -> str:
    """
    tido sample_id (GSE208253_GSM6339631) →
    내 모델 sample_id (GSM6339631_s1) 로 정규화.
    공통 key를 만들어 두 모델의 sample을 매핑.
    """
    # tido 형태: GSE208253_GSM6339631, GSE220978_Patient1, Queensland_P5 등
    parts = sid.split('_')
    if parts[0].startswith('GSE') or parts[0] in ('Zenodo', 'Queensland'):
        return '_'.join(parts[1:])
    return sid


def sort_samples(sids):
    def key(s):
        ds = get_dataset(s)
        rank = DATASET_ORDER.index(ds) if ds in DATASET_ORDER else len(DATASET_ORDER)
        return (rank, s)
    return sorted(sids, key=key)


def make_palette(sids):
    """정규화된 sample_id 기준으로 색상 할당 (두 모델에서 같은 색)"""
    palette = []
    for name in ['tab20', 'tab20b', 'tab20c']:
        cm = plt.get_cmap(name)
        palette += [cm(i) for i in range(cm.N)]
    return {sid: palette[i % len(palette)] for i, sid in enumerate(sids)}


# ────────────────────────────────────────────────────────────
# 내 모델 데이터 로드 (fold별 npy merge)
# ────────────────────────────────────────────────────────────
def load_my_model(emb_dir: Path, npy_name: str) -> dict:
    merged = defaultdict(list)
    found = []
    for fold_dir in sorted(emb_dir.glob('fold_*')):
        npy_path = fold_dir / npy_name
        if not npy_path.exists():
            print(f"  [skip] {npy_path}")
            continue
        data = np.load(npy_path, allow_pickle=True).item()
        found.append(fold_dir.name)
        for sid, arr in data.items():
            merged[sid].append(arr)
    print(f"  Loaded folds: {found}")
    return {sid: np.concatenate(arrs) for sid, arrs in merged.items()}


# ────────────────────────────────────────────────────────────
# Violin 그리기
# ────────────────────────────────────────────────────────────
def draw_panel(ax, sids, data, palette, title, shared_ylim=None):
    """
    sids      : 정렬된 sample_id (내 모델 기준)
    data      : {sample_id: np.array}
    palette   : {sample_id: color} — 두 panel 공통
    """
    n      = len(sids)
    pos    = list(range(1, n + 1))
    vdata  = [data[s] for s in sids]
    colors = [palette[s] for s in sids]

    vp = ax.violinplot(vdata, positions=pos,
                       showmeans=True, showmedians=True,
                       showextrema=True, widths=0.75)

    for body, c in zip(vp['bodies'], colors):
        body.set_facecolor(c)
        body.set_alpha(0.72)
        body.set_edgecolor('black')
        body.set_linewidth(0.5)

    for k in ('cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxes'):
        if k in vp:
            vp[k].set_color('black')
            vp[k].set_linewidth(1.6 if k == 'cmeans' else 1.0)
            if k == 'cmeans':
                vp[k].set_linestyle('--')

    # 데이터셋 경계선
    prev_ds = None
    for i, s in enumerate(sids):
        ds = get_dataset(s)
        if prev_ds is not None and ds != prev_ds:
            ax.axvline(i + 0.5, color='#888', linestyle='--',
                       linewidth=1.0, alpha=0.55)
            ax.text(i + 0.55, (shared_ylim[1] if shared_ylim else ax.get_ylim()[1]) * 0.97,
                    ds, ha='left', va='top', fontsize=7,
                    color='#444', fontstyle='italic')
        prev_ds = ds

    # 배경 홀짝 shading
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

    # mean 값 상단
    ylim_top = shared_ylim[1] if shared_ylim else ax.get_ylim()[1]
    for p, s in zip(pos, sids):
        ax.text(p, ylim_top, f"{data[s].mean():.3f}",
                ha='center', va='bottom', fontsize=5.5,
                color='#111', fontweight='bold')

    ax.axhline(0, color='red', linestyle=':', linewidth=0.9, alpha=0.6)
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.set_xticks(pos)
    ax.set_xticklabels(sids, rotation=55, ha='right', fontsize=7)
    ax.set_xlim(0.2, n + 0.8)
    ax.set_ylabel('Gene-wise PCC', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=6)

    if shared_ylim:
        ax.set_ylim(*shared_ylim)

    # 범례
    handles = [
        mpatches.Patch(facecolor=palette[s], label=s,
                       alpha=0.85, edgecolor='black', linewidth=0.4)
        for s in sids
    ]
    ax.legend(handles=handles,
              loc='upper left', bbox_to_anchor=(1.01, 1.0),
              fontsize=7, framealpha=0.9, edgecolor='#ccc',
              ncol=1, handlelength=1.2, labelspacing=0.3,
              borderpad=0.6)


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────
def main(args):

    # ── 내 모델 로드 ──────────────────────────────────────────
    print("Loading my model results...")
    my_data_raw = load_my_model(Path(args.my_emb_dir), args.my_npy_name)
    print(f"  My model samples: {len(my_data_raw)}")

    # ── Tido 모델 로드 ────────────────────────────────────────
    print("\nLoading tido model results...")
    tido_data_raw = np.load(args.tido_npy, allow_pickle=True).item()
    print(f"  Tido samples: {len(tido_data_raw)}")

    # ── sample_id 정규화 (tido → 내 모델 기준) ───────────────
    # tido: GSE208253_GSM6339631 → GSM6339631  (단, s1 suffix가 없으면 못 매핑)
    # 내 모델: GSM6339631_s1
    # → 정규화: 두 모델 모두 "핵심 ID"로 매핑
    #
    # 전략: 두 모델에서 공통으로 존재하는 샘플만 비교
    #       내 모델 sample_id를 기준으로 tido sample_id를 역매핑

    # 내 모델 sid → tido sid 매핑 테이블
    # tido sid에서 GSE prefix 제거한 값이 내 모델 sid와 매칭되어야 함
    tido_core = {}   # core_id → tido_sid
    for tsid in tido_data_raw:
        parts = tsid.split('_')
        if parts[0].startswith('GSE') or parts[0] in ('Zenodo', 'Queensland'):
            core = '_'.join(parts[1:])
        else:
            core = tsid
        # Queensland / Zenodo 특수 처리
        if 'P5' in tsid and 'Data' not in tsid:
            core = 'P5'
        elif 'Visium' in tsid:
            core = 'Visium_S01'
        elif '17B5776' in tsid:
            core = '17B5776'
        elif '19h1257' in tsid:
            core = '19h1257'
        tido_core[core] = tsid

        # GSE281978: GSM8633892_21_00758_LI_SING → 단축형 GSM8633892 도 등록
        if 'LI_SING' in tsid:
            gsm_short = core.split('_')[0]   # GSM8633892
            if gsm_short not in tido_core:
                tido_core[gsm_short] = tsid

    # 내 모델 sid → tido sid 매핑
    my_to_tido = {}
    for msid in my_data_raw:
        candidates = [msid]
        parts = msid.split('_')
        # GSM6339631_s1 → GSM6339631 (s suffix 제거)
        if len(parts) >= 2 and parts[-1].startswith('s') and parts[-1][1:].isdigit():
            candidates.append('_'.join(parts[:-1]))
        # GSM8633892_21_00758_LI_SING → GSM8633892 (LI_SING 계열 단축)
        if 'LI_SING' in msid:
            candidates.append(parts[0])   # GSM8633892 만
        for cand in candidates:
            if cand in tido_core:
                my_to_tido[msid] = tido_core[cand]
                break

    print("\n  Sample mapping (my → tido):")
    for msid, tsid in sorted(my_to_tido.items()):
        print(f"    {msid:<40} → {tsid}")

    unmapped = [s for s in my_data_raw if s not in my_to_tido]
    if unmapped:
        print(f"\n  ⚠️  매핑 실패 (내 모델): {unmapped}")

    # 공통 샘플만 사용
    common_my_sids = sort_samples([s for s in my_data_raw if s in my_to_tido])

    my_data   = {s: my_data_raw[s]                        for s in common_my_sids}
    tido_data = {s: tido_data_raw[my_to_tido[s]]          for s in common_my_sids}

    n = len(common_my_sids)
    print(f"\n  Common samples for comparison: {n}")

    # ── 공통 palette (내 모델 sample_id 기준) ────────────────
    palette = make_palette(common_my_sids)

    shared_ylim = (-0.4, 1.0)

    # ── Figure ───────────────────────────────────────────────
    fw  = max(20, n * 0.65)
    fh  = 12.0   # 2 panels
    fig, axes = plt.subplots(2, 1, figsize=(fw, fh))

    # 콘솔 비교 요약
    print(f"\n{'Sample':<40} {'My mean':>9} {'Tido mean':>10} {'Diff':>8}")
    print("-" * 70)
    for s in common_my_sids:
        my_m   = my_data[s].mean()
        tido_m = tido_data[s].mean()
        print(f"{s:<40} {my_m:>9.4f} {tido_m:>10.4f} {my_m-tido_m:>+8.4f}")

    # Panel 1: 내 모델
    draw_panel(axes[0], common_my_sids, my_data, palette,
               title="My Model (OmiCLIP Fine-tuned)  ·  Gene-wise PCC per Sample  (Top 300 genes)",
               shared_ylim=shared_ylim)

    # Panel 2: Tido 모델
    draw_panel(axes[1], common_my_sids, tido_data, palette,
               title="Tido Model  ·  Gene-wise PCC per Sample  (Top 300 genes)",
               shared_ylim=shared_ylim)

    plt.tight_layout()
    plt.subplots_adjust(right=0.80, hspace=0.55)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved: {out}")
    plt.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description="내 모델 vs Tido 모델 gene-wise PCC violin 비교"
    )
    ap.add_argument('--my_emb_dir',  required=True,
                    help='fold_01~fold_10 디렉터리가 있는 상위 경로')
    ap.add_argument('--my_npy_name', default='Top_500_sample_gene_pcc_dist.npy',
                    help='내 모델 각 fold의 npy 파일명')
    ap.add_argument('--tido_npy',    required=True,
                    help='violin_tido.py 가 생성한 tido_sample_gene_pcc_dist.npy 경로')
    ap.add_argument('--output',      default='compare_violin.png')
    main(ap.parse_args())