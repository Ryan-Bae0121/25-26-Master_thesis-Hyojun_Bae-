#!/usr/bin/env python3
"""
Fine-tuned violin plot - zeroshot_violin.py 와 완전히 동일한 스타일
fold_01~fold_10 의 Top_500_sample_gene_pcc_dist.npy 를 읽어 샘플별로 합산
"""
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DATASET_ORDER = ['GSE181300','GSE208253','GSE220978','GSE252265','GSE281978','Queensland','Zenodo']
QUEENSLAND_PREFIXES = ['visium_s','p5']
GSM_TO_GSE = {
    'GSM5494475':'GSE181300','GSM5494476':'GSE181300','GSM5494477':'GSE181300',
    'GSM5494478':'GSE181300','GSM5494479':'GSE181300','GSM5494480':'GSE181300',
    'GSM5494481':'GSE181300','GSM5494482':'GSE181300',
    'GSM6339631_s1':'GSE208253','GSM6339632_s2':'GSE208253','GSM6339633_s3':'GSE208253',
    'GSM6339634_s4':'GSE208253','GSM6339635_s5':'GSE208253','GSM6339636_s6':'GSE208253',
    'GSM6339637_s7':'GSE208253','GSM6339638_s8':'GSE208253','GSM6339639_s9':'GSE208253',
    'GSM6339640_s10':'GSE208253','GSM6339641_s11':'GSE208253','GSM6339642_s12':'GSE208253',
    'GSM7998252':'GSE252265','GSM7998253':'GSE252265','GSM7998254':'GSE252265',
    'GSM7998255':'GSE252265','GSM7998256':'GSE252265','GSM7998257':'GSE252265',
    'GSM7998258':'GSE252265','GSM7998259':'GSE252265',
    'GSM8633891_21_00757_LI_SING':'GSE281978','GSM8633892_21_00758_LI_SING':'GSE281978',
    'GSM8633892_21_00759_LI_SING':'GSE281978','GSM8633893_21_01569_LI_SING':'GSE281978',
    'GSM8633894_21_01570_LI_SING':'GSE281978','GSM8633895_21_01586_LI_SING':'GSE281978',
    'GSM8633896_21_01587_LI_SING':'GSE281978',
}

DS_BG = {
    'GSE181300':  '#AED6F1',
    'GSE208253':  '#A9DFBF',
    'GSE220978':  '#F9E79F',
    'GSE252265':  '#F5CBA7',
    'GSE281978':  '#D2B4DE',
    'Queensland': '#AEB6BF',
    'Zenodo':     '#F1948A',
}

def get_dataset(sid):
    if sid in GSM_TO_GSE: return GSM_TO_GSE[sid]
    sl = sid.lower()
    if sl in ('patient1','patient2','patient3','patient4'): return 'GSE220978'
    for ds in DATASET_ORDER:
        if ds.lower() in sl: return ds
    if any(sl.startswith(p) for p in QUEENSLAND_PREFIXES): return 'Queensland'
    if sid[0].isdigit(): return 'Zenodo'
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


def draw_violin(ax, sids, data, title, pcc_type='gene'):
    n        = len(sids)
    pos      = list(range(1, n + 1))
    vdata    = [data[s] for s in sids]
    palette  = make_palette(sids)
    colors   = [palette[s] for s in sids]

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

    # GSE181300 첫 레이블
    first_ds = get_dataset(sids[0])
    ax.text(0.52, ymax * 0.97, first_ds,
            ha='left', va='top', fontsize=7,
            color='#444', fontstyle='italic')

    # 데이터셋 경계 점선 + 나머지 레이블
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

    # mean 값 상단 표시
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


# ── 데이터 로드 (violin_pcc_merged.py 방식 - .item() dict)
emb_dir = Path('/project_antwerp/hbae/Loki_output/0228_10fold_finetune_embedding')
merged = defaultdict(list)
found_folds = []

for fold_dir in sorted(emb_dir.glob('fold_*')):
    npy_path = fold_dir / 'Top_500_sample_gene_pcc_dist.npy'
    if not npy_path.exists():
        print(f"  [skip] {npy_path}")
        continue
    fold_data = np.load(npy_path, allow_pickle=True).item()
    found_folds.append(fold_dir.name)
    for sid, arr in fold_data.items():
        merged[sid].append(arr)

print(f"Loaded folds: {found_folds}")
print(f"Total samples: {len(merged)}")

data = {sid: np.concatenate(arrs) for sid, arrs in merged.items()}
sids = sort_samples(list(data.keys()))

n   = len(sids)
fw  = max(16, n * 0.65)
fig, ax = plt.subplots(figsize=(fw, 6.5))

title = (f"Gene-wise PCC per Sample  "
         f"(Top 300 genes · Fine-tuned · {len(found_folds)} folds · {n} samples)")
draw_violin(ax, sids, data, title)

plt.tight_layout()
plt.subplots_adjust(right=0.78)
out = Path('/project_antwerp/hbae/script/0208_start/finetune_violin.png')
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved: {out}")