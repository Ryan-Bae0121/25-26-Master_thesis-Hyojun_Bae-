#!/usr/bin/env python3
"""
violin_tido_31.py
=================
31개 샘플 (Patient1~4, 19h1257 제외) TIDO 예측 결과 분석.
tido_prediction_results.csv 의 예측값과
combined_expression_matrix.npy 의 ground truth를 spot 단위로 매칭해
샘플별 gene-wise PCC 분포를 violin plot으로 시각화.

★ top 300 most highly expressed genes (val set 기준, Loki 논문 방식)
★ 각 샘플 고유 색상
★ 데이터셋 순서: GSE181300 → GSE208253 → GSE252265 → GSE281978 → Queensland → Zenodo

Usage:
    python violin_tido_31.py \
        --pred  /project_antwerp/hbae/data/TIDO/tido_prediction_results_31samples.csv \
        --expr  /project_antwerp/hbae/data/0317_HVG_NEW/combined_expression_matrix.npy \
        --obs   /project_antwerp/hbae/data/0317_HVG_NEW/combined_obs.npy \
        --genes /project_antwerp/hbae/data/0317_hvg_2000_list.txt \
        --output /project_antwerp/hbae/script/0208_start/Violin_plot/tido_31s_violin.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm


# ────────────────────────────────────────────────────────────
# Dataset 정의  (GSE220978 제외 — Patient 샘플들)
# ────────────────────────────────────────────────────────────
DATASET_ORDER = [
    'GSE181300', 'GSE208253', 'GSE252265', 'GSE281978', 'Queensland', 'Zenodo',
]

GSM_TO_GSE = {
    # GSE181300
    'GSM5494475': 'GSE181300', 'GSM5494476': 'GSE181300',
    'GSM5494477': 'GSE181300', 'GSM5494478': 'GSE181300',
    'GSM5494479': 'GSE181300', 'GSM5494480': 'GSE181300',
    'GSM5494481': 'GSE181300', 'GSM5494482': 'GSE181300',
    # GSE208253
    'GSM6339631': 'GSE208253', 'GSM6339632': 'GSE208253',
    'GSM6339633': 'GSE208253', 'GSM6339634': 'GSE208253',
    'GSM6339635': 'GSE208253', 'GSM6339636': 'GSE208253',
    'GSM6339637': 'GSE208253', 'GSM6339638': 'GSE208253',
    'GSM6339639': 'GSE208253', 'GSM6339640': 'GSE208253',
    'GSM6339641': 'GSE208253', 'GSM6339642': 'GSE208253',
    # GSE252265
    'GSM7998252': 'GSE252265', 'GSM7998253': 'GSE252265',
    'GSM7998254': 'GSE252265', 'GSM7998255': 'GSE252265',
    'GSM7998256': 'GSE252265', 'GSM7998257': 'GSE252265',
    'GSM7998258': 'GSE252265', 'GSM7998259': 'GSE252265',
    # GSE281978
    'GSM8633891': 'GSE281978', 'GSM8633892': 'GSE281978',
    'GSM8633893': 'GSE281978', 'GSM8633894': 'GSE281978',
    'GSM8633895': 'GSE281978', 'GSM8633896': 'GSE281978',
}

# 제외된 샘플 (Patient1~4, 19h1257)
EXCLUDED_SAMPLES = {'Patient1', 'Patient2', 'Patient3', 'Patient4', '19h1257'}

DS_BG = {
    'GSE181300':  '#AED6F1',
    'GSE208253':  '#A9DFBF',
    'GSE220978':  '#F9E79F',
    'GSE252265':  '#F5CBA7',
    'GSE281978':  '#D2B4DE',
    'Queensland': '#AEB6BF',
    'Zenodo':     '#F1948A',
}


def get_dataset(sid: str) -> str:
    # GSE181300_GSM... 형태 — 앞 GSE prefix로 바로 판별
    for ds in DATASET_ORDER:
        if sid.startswith(ds):
            return ds
    # GSM prefix만 있는 경우
    for gsm_prefix, gse in GSM_TO_GSE.items():
        if sid.startswith(gsm_prefix):
            return gse
    sl = sid.lower()
    if 'queensland' in sl or 'visium' in sl or sl.startswith('p5'):
        return 'Queensland'
    if sid[0].isdigit():
        return 'Zenodo'
    return 'Unknown'


def sort_samples(sids):
    def key(s):
        ds = get_dataset(s)
        rank = DATASET_ORDER.index(ds) if ds in DATASET_ORDER else len(DATASET_ORDER)
        return (rank, s)
    return sorted(sids, key=key)


def make_palette(sids):
    palette = []
    for name in ['tab20', 'tab20b', 'tab20c']:
        cm = plt.get_cmap(name)
        palette += [cm(i) for i in range(cm.N)]
    return {sid: palette[i % len(palette)] for i, sid in enumerate(sids)}


# ────────────────────────────────────────────────────────────
# PCC 계산
# ────────────────────────────────────────────────────────────
def calc_gene_pcc(preds: np.ndarray, exprs: np.ndarray) -> np.ndarray:
    corrs = []
    for g in range(preds.shape[1]):
        if exprs[:, g].std() > 1e-8:
            r, _ = pearsonr(preds[:, g], exprs[:, g])
            if np.isfinite(r):
                corrs.append(r)
    return np.array(corrs)


# ────────────────────────────────────────────────────────────
# Violin
# ────────────────────────────────────────────────────────────
def draw_violin(ax, sids, data, title, pcc_type='gene'):
    n       = len(sids)
    pos     = list(range(1, n + 1))
    vdata   = [data[s] for s in sids]
    palette = make_palette(sids)
    colors  = [palette[s] for s in sids]

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

    ax.set_ylim(-0.4, 1.0)
    ymax = ax.get_ylim()[1]

    # 첫 dataset 레이블
    first_ds = get_dataset(sids[0])
    ax.text(0.52, ymax * 0.97, first_ds,
            ha='left', va='top', fontsize=7,
            color='#444', fontstyle='italic')

    # 데이터셋 경계 점선 + 레이블
    prev_ds = None
    for i, s in enumerate(sids):
        ds = get_dataset(s)
        if prev_ds is not None and ds != prev_ds:
            ax.axvline(i + 0.5, color='#888', linestyle='--',
                       linewidth=1.0, alpha=0.55)
            ax.text(i + 0.52, ymax * 0.97,
                    ds, ha='left', va='top', fontsize=7,
                    color='#444', fontstyle='italic')
        prev_ds = ds

    # DS_BG 배경 shading
    prev_ds, ds_start = None, 1
    for i, s in enumerate(sids):
        ds = get_dataset(s)
        if prev_ds is not None and ds != prev_ds:
            ax.axvspan(ds_start - 0.5, i + 0.5,
                       color=DS_BG.get(prev_ds, '#eeeeee'), alpha=0.25, zorder=0)
            ds_start = i + 1
        prev_ds = ds
    ax.axvspan(ds_start - 0.5, n + 0.5,
               color=DS_BG.get(prev_ds, '#eeeeee'), alpha=0.25, zorder=0)

    for p, s in zip(pos, sids):
        ax.text(p, ymax, f"{data[s].mean():.3f}",
                ha='center', va='bottom', fontsize=6,
                color='#111', fontweight='bold')

    ax.set_xticks(pos)
    ax.set_xticklabels(sids, rotation=55, ha='right', fontsize=7.5)
    ax.set_xlim(0.2, n + 0.8)
    ylabel = 'Gene-wise PCC' if pcc_type == 'gene' else 'Spot-wise PCC'
    ax.set_ylabel(ylabel, fontsize=11)
    ax.axhline(0, color='red', linestyle=':', linewidth=0.9, alpha=0.6)
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=8)

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
# Main
# ────────────────────────────────────────────────────────────
def main(args):

    # ── 1. tido prediction 로드 ───────────────────────────────
    print("Loading tido predictions...")
    pred_df = pd.read_csv(args.pred)
    # 첫 컬럼이 tile_id (Unnamed: 0 또는 tile_id)
    first_col = pred_df.columns[0]
    if first_col != 'tile_id':
        pred_df = pred_df.rename(columns={first_col: 'tile_id'})
    print(f"  shape: {pred_df.shape}")
    print(f"  tile_id sample: {pred_df['tile_id'].iloc[0]}")

    def extract_sample_id(tile_id):
        if tile_id.startswith('Queensland P5'):
            return 'Queensland_P5'
        if 'Visium' in tile_id:
            parts = tile_id.split('_')
            # Queensland_Visium_S01_... → Queensland_Visium_S01
            if parts[0] == 'Queensland' and parts[1] == 'Visium':
                return f"Queensland_Visium_{parts[2]}"
        parts = tile_id.split('_')
        return '_'.join(parts[:2])

    pred_df['sample_id'] = pred_df['tile_id'].apply(extract_sample_id)

    # 제외된 샘플 필터링
    def is_excluded(sid):
        # GSE220978_Patient1 → Patient1 추출
        parts = sid.split('_')
        core = '_'.join(parts[1:]) if parts[0].startswith('GSE') else sid
        return core in EXCLUDED_SAMPLES or any(e in sid for e in EXCLUDED_SAMPLES)

    before = len(pred_df)
    pred_df = pred_df[~pred_df['sample_id'].apply(is_excluded)].copy()
    print(f"  Excluded Patient1~4 & 19h1257: {before:,} → {len(pred_df):,} spots")

    tido_gene_cols = [c for c in pred_df.columns if c not in ('tile_id', 'sample_id')]
    print(f"  samples: {pred_df['sample_id'].nunique()}")
    print(f"  genes  : {len(tido_gene_cols)}")
    print(f"  unique sample_ids: {sorted(pred_df['sample_id'].unique())}")

    # ── 2. Ground truth 로드 ──────────────────────────────────
    print("\nLoading ground truth...")
    gt_expr = np.load(args.expr)
    gt_obs  = np.load(args.obs, allow_pickle=True)
    print(f"  gt_expr shape: {gt_expr.shape}")
    print(f"  gt_obs  shape: {gt_obs.shape}")

    hvg_genes = open(args.genes).read().splitlines()
    print(f"  HVG genes: {len(hvg_genes)}")

    all_genes_file = Path(args.genes).parent / 'all_shared_genes.txt'
    all_genes = open(all_genes_file).read().splitlines()
    print(f"  all_shared_genes: {len(all_genes)}")
    hvg_idx = np.array([all_genes.index(g) for g in hvg_genes if g in all_genes])
    gt_expr_hvg = gt_expr[:, hvg_idx]
    hvg_genes_matched = [g for g in hvg_genes if g in set(all_genes)]
    print(f"  gt_expr_hvg shape: {gt_expr_hvg.shape}")

    gt_obs_clean = np.array([o.replace('_hires', '') for o in gt_obs])

    # ── 3. spot 매칭 ──────────────────────────────────────────
    print("\nMatching spots...")

    def extract_spot_key(tile_id):
        if tile_id.startswith('Queensland P5'):
            parts = tile_id.split('_')
            return f"P5_{parts[2]}"

        parts = tile_id.split('_')

        if parts[0].startswith('GSE'):
            parts = parts[1:]
        elif parts[0] in ('Zenodo', 'Queensland'):
            parts = parts[1:]

        sample = parts[0]

        if sample.startswith('GSM8633'):
            li_idx = next((i for i, p in enumerate(parts) if p == 'SING'), None)
            if li_idx is not None and len(parts) > li_idx + 1:
                sample_full = '_'.join(parts[:li_idx + 1])
                barcode     = parts[li_idx + 1]
                return f"{sample_full}_{barcode}"

        if sample.startswith('GSM6339'):
            if len(parts) > 2 and parts[1].startswith('s') and parts[1][1:].isdigit():
                return f"{sample}_{parts[1]}_{parts[2]}"

        if sample == 'Visium':
            return f"Visium_{parts[1]}_{parts[2]}" if len(parts) > 2 else f"Visium_{parts[1]}"

        barcode = parts[1] if len(parts) > 1 else ''
        return f"{sample}_{barcode}"

    pred_df['spot_key'] = pred_df['tile_id'].apply(extract_spot_key)

    print("  Building gt obs index...")
    gt_obs_idx = {obs: i for i, obs in enumerate(gt_obs_clean)}

    pred_df['gt_idx'] = pred_df['spot_key'].map(gt_obs_idx)
    matched = pred_df.dropna(subset=['gt_idx']).copy()
    matched['gt_idx'] = matched['gt_idx'].astype(int)

    total    = len(pred_df)
    n_matched = len(matched)
    print(f"  Total spots: {total:,}  →  Matched: {n_matched:,}  "
          f"({n_matched/total*100:.1f}%)")

    if n_matched == 0:
        print("\n❌ 매칭 실패! spot_key 샘플:")
        print("  tido:", pred_df['spot_key'].iloc[:5].tolist())
        print("  gt  :", gt_obs_clean[:5].tolist())
        return

    # ── 4. gene 순서 통일 ─────────────────────────────────────
    tido_gene_set = set(tido_gene_cols)
    reorder_idx = [tido_gene_cols.index(g) for g in hvg_genes_matched
                   if g in tido_gene_set]
    hvg_genes_common = [g for g in hvg_genes_matched if g in tido_gene_set]
    print(f"\n  Common genes (tido ∩ HVG): {len(hvg_genes_common)}")

    # hvg_genes_common에 해당하는 gt_expr_hvg 인덱스
    hvg_matched_set = {g: i for i, g in enumerate(hvg_genes_matched)}
    gt_reorder_idx = np.array([hvg_matched_set[g] for g in hvg_genes_common])

    # ── 5. Fold 구성 (31샘플 기준 — Patient1~4, 19h1257 제외) ──
    # 원래 fold에서 제외 샘플 빼고 재구성
    FOLD_SAMPLES = {
        'fold_01': ['GSM6339631_s1',               'GSM8633892_21_00758_LI_SING'],
        'fold_02': ['GSM6339635_s5',               'GSM7998258'],
        'fold_03': ['GSM5494476',                  'GSM5494477'],
        'fold_04': ['GSM6339632_s2',               'GSM8633895_21_01586_LI_SING'],
        'fold_05': ['GSM7998252',                  'GSM8633893_21_01569_LI_SING',
                    'GSM8633894_21_01570_LI_SING', 'P5'],
        'fold_06': ['GSM6339633_s3',               'GSM6339634_s4',
                    'GSM6339640_s10',              'GSM8633896_21_01587_LI_SING'],
        'fold_07': ['GSM5494478',                  'GSM6339641_s11',
                    'GSM7998257',                  'GSM8633891_21_00757_LI_SING'],
        'fold_08': ['GSM6339642_s12',              'GSM7998254',
                    'GSM7998256',                  'Visium_S01'],
        'fold_09': ['GSM6339637_s7',               'GSM7998253',
                    'GSM7998259'],
        'fold_10': ['17B5776',                     'GSM5494475',
                    'GSM6339638_s8',               'GSM7998255'],
    }

    def match_fold(tido_sid):
        parts = tido_sid.split('_')
        if parts[0].startswith('GSE') or parts[0] in ('Zenodo', 'Queensland'):
            core = '_'.join(parts[1:])
        else:
            core = tido_sid

        if 'P5' in tido_sid and 'Data' not in tido_sid:
            core = 'P5'
        elif 'Visium' in tido_sid:
            core = 'Visium_S01'
        elif '17B5776' in tido_sid:
            core = '17B5776'

        for fold, samples in FOLD_SAMPLES.items():
            if core in samples:
                return fold

        for suffix in [f'_s{i}' for i in range(1, 13)]:
            candidate = core + suffix
            for fold, samples in FOLD_SAMPLES.items():
                if candidate in samples:
                    return fold

        GSM_FULL = {
            'GSM8633891': 'GSM8633891_21_00757_LI_SING',
            'GSM8633892': 'GSM8633892_21_00758_LI_SING',
            'GSM8633893': 'GSM8633893_21_01569_LI_SING',
            'GSM8633894': 'GSM8633894_21_01570_LI_SING',
            'GSM8633895': 'GSM8633895_21_01586_LI_SING',
            'GSM8633896': 'GSM8633896_21_01587_LI_SING',
        }
        if core in GSM_FULL:
            full = GSM_FULL[core]
            for fold, samples in FOLD_SAMPLES.items():
                if full in samples:
                    return fold

        return None

    print("\n  Sample → Fold mapping:")
    unique_sids  = matched['sample_id'].unique()
    sid_to_fold  = {}
    for sid in sorted(unique_sids):
        fold = match_fold(sid)
        sid_to_fold[sid] = fold
        print(f"    {sid:<45} → {fold}")

    unmatched = [s for s, f in sid_to_fold.items() if f is None]
    if unmatched:
        print(f"\n  ⚠️  fold 매핑 실패: {unmatched}")

    # ── 6. 샘플별 gene-wise PCC ───────────────────────────────
    print("\nComputing gene-wise PCC per sample (top 300)...")
    sample_gene_pccs = {}

    for fold_name in FOLD_SAMPLES:
        fold_tido_sids = [s for s, f in sid_to_fold.items() if f == fold_name]
        if not fold_tido_sids:
            continue

        fold_matched = matched[matched['sample_id'].isin(fold_tido_sids)]
        fold_gt_idx  = fold_matched['gt_idx'].values.astype(int)
        fold_mean    = gt_expr_hvg[fold_gt_idx][:, gt_reorder_idx].mean(axis=0)
        top300_local = np.argsort(fold_mean)[::-1][:300]

        print(f"\n  [{fold_name}] top 300 from {len(fold_gt_idx):,} spots "
              f"({len(fold_tido_sids)} samples)")

        for sid in fold_tido_sids:
            group      = matched[matched['sample_id'] == sid]
            gt_indices = group['gt_idx'].values.astype(int)
            pred_vals  = group[tido_gene_cols].values[:, reorder_idx]  # → hvg_genes_common 순서
            gt_vals    = gt_expr_hvg[gt_indices][:, gt_reorder_idx]

            pred_top = pred_vals[:, top300_local]
            gt_top   = gt_vals[:, top300_local]

            gene_pccs = calc_gene_pcc(pred_top, gt_top)
            sample_gene_pccs[sid] = gene_pccs

            print(f"    {sid:<45} n_spots={len(group):>5}  "
                  f"n_genes={len(gene_pccs):>3}  "
                  f"mean_PCC={gene_pccs.mean():.4f}")

    # ── 7. npy 저장 ───────────────────────────────────────────
    out_npy = Path(args.output).parent / 'tido_31sample_gene_pcc_dist.npy'
    np.save(out_npy, sample_gene_pccs)
    print(f"\n✅ Saved: {out_npy}")

    # ── 8. Violin plot ────────────────────────────────────────
    sids = sort_samples(list(sample_gene_pccs.keys()))
    n    = len(sids)
    fw   = max(18, n * 0.65)
    fh   = 6.5

    fig, ax = plt.subplots(figsize=(fw, fh))
    title = (f"Gene-wise PCC per Sample  "
             f"(Top 300 genes · Tido Model · {n} samples)")
    draw_violin(ax, sids, sample_gene_pccs, title)

    plt.tight_layout()
    plt.subplots_adjust(right=0.78)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"✅ Saved violin plot: {out}")
    plt.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred',   required=True)
    ap.add_argument('--expr',   required=True)
    ap.add_argument('--obs',    required=True)
    ap.add_argument('--genes',  required=True)
    ap.add_argument('--output', default='/tmp/tido_violin.png')
    main(ap.parse_args())