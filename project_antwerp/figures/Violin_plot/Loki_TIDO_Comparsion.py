#!/usr/bin/env python3
"""
LOKI vs TIDO comparison (8 violins, 2 colors)

Groups (x-axis): All genes, HEG top-300, HVG top-300, Top PCC (oracle)
Within each group: [LOKI, TIDO]
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE = Path("/project_antwerp/hbae/Loki_output")

# ---- LOKI: fold concat sources (same as your original)
LOKI_METHODS = [
    ("All genes",        BASE / "openst_validation_agg_v2", "openst_allgene_pcc.npy"),
    ("HEG top-300",      BASE / "openst_validation_agg_v2", "openst_genewise_pcc.npy"),
    ("HVG top-300",      BASE / "openst_validation_hvg300", "openst_genewise_pcc.npy"),
    ("Top PCC (oracle)", BASE / "openst_validation_agg_v2", "openst_genewise_pcc_toppcc.npy"),
]

def load_loki_concat(d: Path, fname: str) -> np.ndarray:
    arrs = []
    for fold_dir in sorted(d.glob("fold_*")):
        p = fold_dir / fname
        if p.exists():
            arrs.append(np.load(p))
    if not arrs:
        return np.array([])
    return np.concatenate(arrs)

# ---- TIDO: load precomputed arrays from npz (saved from your TIDO script)
TIDO_NPZ = Path("/project_antwerp/hbae/figures/Violin_plot/tido_pcc_4methods.npz")
tido = np.load(TIDO_NPZ)
TIDO_METHODS = [
    ("All genes",        tido["all_genes"]),
    ("HEG top-300",      tido["heg"]),
    ("HVG top-300",      tido["hvg"]),
    ("Top PCC-300(oracle)", tido["oracle"]),
]

# ---- Colors: only 2
COLOR_LOKI = "#4C72B0"
COLOR_TIDO = "#DD8452"

# ---- Assemble data (8 violins)
group_labels = [x[0] for x in LOKI_METHODS]  # 4 groups
data = []
colors = []
positions = []

group_x = np.arange(1, 5)         # 1..4
dx = 0.18                          # left/right offset within group

for i, gx in enumerate(group_x):
    loki_arr = load_loki_concat(LOKI_METHODS[i][1], LOKI_METHODS[i][2])
    tido_arr = TIDO_METHODS[i][1]

    # order: LOKI (left), TIDO (right)
    data.extend([loki_arr, tido_arr])
    colors.extend([COLOR_LOKI, COLOR_TIDO])
    positions.extend([gx - dx, gx + dx])

# ---- Plot
fig, ax = plt.subplots(figsize=(10, 6))

vp = ax.violinplot(
    data,
    positions=positions,
    showmeans=True,
    showmedians=True,
    showextrema=True,
    widths=0.30,
)

for body, c in zip(vp["bodies"], colors):
    body.set_facecolor(c)
    body.set_alpha(0.80)
    body.set_edgecolor("black")
    body.set_linewidth(0.8)

for k in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
    if k in vp:
        vp[k].set_color("black")
        vp[k].set_linewidth(1.6 if k == "cmeans" else 1.0)
        if k == "cmeans":
            vp[k].set_linestyle("--")

ax.set_ylim(-0.4, 1.0)
ax.axhline(0, color="red", linestyle=":", linewidth=0.9, alpha=0.6)
ax.grid(axis="y", linestyle="--", alpha=0.35)

ax.set_xticks(group_x)
ax.set_xticklabels(group_labels, fontsize=10)
ax.set_xlim(0.5, 4.5)
ax.set_ylabel("Gene-wise PCC", fontsize=12)

# mean text above each violin (optional)
ymax = ax.get_ylim()[1]
for x, arr in zip(positions, data):
    if len(arr) == 0:
        continue
    ax.text(x, ymax, f"{np.mean(arr):.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#111")

ax.set_title(
    "Open-ST HNSCC External Validation\nGene-wise PCC — LOKI vs TIDO (4 schemes)",
    fontsize=12, fontweight="bold", pad=20
)

# Legend (2 colors only)
handles = [
    mpatches.Patch(facecolor=COLOR_LOKI, label="LOKI", alpha=0.85, edgecolor="black", linewidth=0.5),
    mpatches.Patch(facecolor=COLOR_TIDO, label="TIDO", alpha=0.85, edgecolor="black", linewidth=0.5),
]
ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.9, edgecolor="#ccc")

plt.tight_layout()
out = Path("/project_antwerp/hbae/figures/Violin_plot/openst_loki_vs_tido_8violin.png")
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")