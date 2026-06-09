#!/usr/bin/env python3
"""
TCGA tile PCC visualization on WSI (lowres) + grid heatmaps.

Environment (examples):
  FIXED_SLIDES=TCGA-MZ-A7D7-01Z-00-DX1,TCGA-CR-6472-01Z-00-DX1,TCGA-F7-A61V-01Z-00-DX1
  HDF5_KEY_ORDER=x_y          # HDF5 key = x_topleft_y_topleft at level 0
  CROP_PX_LEVEL0=512            # box size on slide before resize to 256
  SORT_HDF5_KEYS=0            # match `for k in f.keys()` embedding export (no sort)

Paths default to /project_antwerp/...; override with env or edit CONFIG below.
"""
from __future__ import annotations

import glob
import os
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn.functional as F
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Paths — override with environment variables or edit
# ---------------------------------------------------------------------------
def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


GENE_LIST = _env("GENE_LIST", "/project_antwerp/hbae/data/0317_hvg_2000_list.txt")
REF_FILE = _env("REF_FILE", "/project_antwerp/hbae/ref_file.csv")
FT_EMB = _env(
    "FT_EMB",
    "/project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new/fold_03",
)
TCGA_EMB = _env(
    "TCGA_EMB",
    "/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/TCGA_embeddings/fold_03",
)
HDF5_DIR = _env("HDF5_DIR", "/project_antwerp/TCGA-HNSC/TCGA_patch")
WSI_DIR = _env("WSI_DIR", "/project_antwerp/hbae/data/WSIs")
OUT_DIR = _env(
    "OUT_DIR",
    "/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/wsi_visualization",
)

DEVICE = _env("DEVICE", "cuda")
SORT_HDF5_KEYS = os.environ.get("SORT_HDF5_KEYS", "1") not in ("0", "false", "False")

# Comma-separated slide_id (e.g. TCGA-MZ-A7D7,TCGA-CR-6472). Empty = auto-select by tile count.
FIXED_SLIDES = [s.strip() for s in _env("FIXED_SLIDES", "").split(",") if s.strip()]

# HDF5 keys: "x_y" = first number is x (column), second is y (row) at level-0 top-left.
HDF5_KEY_ORDER = _env("HDF5_KEY_ORDER", "x_y")

# Native crop on slide before resize to 256 (TCGA_patch pipeline).
CROP_PX_LEVEL0 = int(_env("CROP_PX_LEVEL0", "512"))


def main() -> None:
    device = DEVICE if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable; using CPU.", file=sys.stderr)
        device = "cpu"

    os.makedirs(OUT_DIR, exist_ok=True)

    with open(GENE_LIST) as f:
        gene_list = [ln.strip() for ln in f if ln.strip()]

    ref_df = pd.read_csv(REF_FILE, index_col=0)
    ref_df["slide_id"] = ref_df["wsi_file_name"].apply(lambda x: str(x).split(".")[0])

    rna_cols = [c for c in ref_df.columns if c.startswith("rna_")]
    ref_genes = [c.replace("rna_", "") for c in rna_cols]
    common_genes = [g for g in gene_list if g in ref_genes]
    common_idx = [gene_list.index(g) for g in common_genes]
    bulk_cols = ["rna_" + g for g in common_genes]
    G = len(common_genes)

    train_embs = F.normalize(
        torch.tensor(
            np.load(f"{FT_EMB}/train_img_embs.npy"),
            dtype=torch.float32,
            device=device,
        ),
        dim=-1,
    )
    train_expr = torch.tensor(
        np.load(f"{FT_EMB}/train_exprs.npy"),
        dtype=torch.float32,
        device=device,
    )
    N_ST = len(train_embs)
    K_30 = int(N_ST * 0.30)

    # WSI path map
    wsi_map: dict[str, str] = {}
    for fpath in glob.glob(f"{WSI_DIR}/*.svs"):
        sid = os.path.basename(fpath).split(".")[0]
        wsi_map[sid] = fpath

    def one_slide(row, sid: str) -> tuple | None:
        hdf5_path = f"{HDF5_DIR}/{sid}/{sid}.hdf5"
        emb_path = f"{TCGA_EMB}/{sid}.npy"
        if not (
            os.path.exists(emb_path)
            and os.path.exists(hdf5_path)
            and sid in wsi_map
        ):
            return None
        n = int(np.load(emb_path, mmap_mode="r").shape[0])
        return (
            sid,
            row[bulk_cols].values.astype(float),
            wsi_map[sid],
            hdf5_path,
            n,
        )

    selected: list[tuple] = []
    if FIXED_SLIDES:
        row_by_sid = {str(r["slide_id"]): r for _, r in ref_df.iterrows()}
        for sid in FIXED_SLIDES:
            if sid not in row_by_sid:
                print(f"WARNING: {sid} not in ref_file.csv", file=sys.stderr)
                continue
            tup = one_slide(row_by_sid[sid], sid)
            if tup is None:
                print(
                    f"WARNING: skip {sid} (missing emb/hdf5/wsi or path mismatch)",
                    file=sys.stderr,
                )
                continue
            selected.append(tup)
        print(f"Selected (FIXED_SLIDES): {[s[0] for s in selected]}")
    else:
        for _, row in ref_df.iterrows():
            sid = row["slide_id"]
            tup = one_slide(row, sid)
            if tup is None:
                continue
            n = tup[4]
            if 1000 <= n <= 8000:
                selected.append(tup)
        selected = selected[:3]
        print(f"Selected (auto): {[s[0] for s in selected]}")

    def get_tile_scores_and_coords(
        sid: str, bulk: np.ndarray, k_spots: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        embs = F.normalize(
            torch.tensor(
                np.load(f"{TCGA_EMB}/{sid}.npy"),
                dtype=torch.float32,
                device=device,
            ),
            dim=-1,
        )
        with torch.no_grad():
            sim_pos = torch.clamp(embs @ train_embs.T, min=0)
            if k_spots:
                topk_v, topk_i = torch.topk(sim_pos, k=min(k_spots, sim_pos.shape[1]), dim=1)
                w = torch.zeros_like(sim_pos)
                w.scatter_(1, topk_i, topk_v / (topk_v.sum(dim=1, keepdim=True) + 1e-8))
            else:
                w = sim_pos / (sim_pos.sum(dim=1, keepdim=True) + 1e-8)
            tp = (w @ train_expr).cpu().numpy()[:, common_idx]

        bulk_c = bulk - bulk.mean()
        tile_c = tp - tp.mean(axis=1, keepdims=True)
        num = (tile_c * bulk_c).sum(axis=1)
        denom = np.sqrt((tile_c**2).sum(axis=1)) * np.sqrt((bulk_c**2).sum())
        scores = np.where(denom > 1e-8, num / denom, -999.0)

        del embs
        if device == "cuda":
            torch.cuda.empty_cache()
        return scores, tp

    def load_hdf5_coords(hdf5_path: str, sort_keys: bool = True) -> tuple[np.ndarray, list[str]]:
        """
        Returns coords (T,2) and keys in the same order as embedding rows.

        Default HDF5_KEY_ORDER=x_y: key "a_b" -> a = x (column), b = y (row) at level 0.
        """
        with h5py.File(hdf5_path, "r") as f:
            keys = list(f.keys())
            if sort_keys:
                keys = sorted(keys)
            rows = []
            for k in keys:
                a, b = k.split("_", 1)
                ia, ib = int(a), int(b)
                if HDF5_KEY_ORDER == "x_y":
                    rows.append([ia, ib])  # x, y
                elif HDF5_KEY_ORDER == "y_x":
                    rows.append([ib, ia])  # store as x,y internally
                else:
                    raise ValueError(f"HDF5_KEY_ORDER must be x_y or y_x, got {HDF5_KEY_ORDER!r}")
            coords = np.array(rows, dtype=np.int64)
        return coords, keys

    def load_wsi_lowres(wsi_path: str):
        """WSI pyramid: level 0 = fullres dimensions for scaling HDF5 coords."""
        tif = tifffile.TiffFile(wsi_path)
        series = tif.series[0]
        fullres_H = int(series.levels[0].shape[0])
        fullres_W = int(series.levels[0].shape[1])

        chosen_level = len(series.levels) - 1
        for li in range(len(series.levels)):
            h = series.levels[li].shape[0]
            if h >= 800:
                chosen_level = li
                break

        img = series.levels[chosen_level].asarray()
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]

        lH, lW = img.shape[:2]
        scale_y = lH / fullres_H
        scale_x = lW / fullres_W
        tif.close()
        print(
            f"    WSI level={chosen_level}: {lH}×{lW}, "
            f"scale=({scale_y:.4f},{scale_x:.4f}), fullres={fullres_H}×{fullres_W}"
        )
        return img, scale_y, scale_x, fullres_H, fullres_W

    # ---------- Figure 1: Violin ----------
    print("\n=== Figure 1: Violin Plot ===")
    fig1, axes = plt.subplots(1, len(selected), figsize=(6 * len(selected), 8))
    if len(selected) == 1:
        axes = [axes]
    fig1.suptitle(
        "Tile-wise PCC Score Distribution\nAll spots (100%) vs K=30% per Slide",
        fontsize=13,
    )

    for ax, (sid, bulk, wsi_path, hdf5_path, n_emb) in zip(axes, selected):
        s_all, _ = get_tile_scores_and_coords(sid, bulk, None)
        s_k30, _ = get_tile_scores_and_coords(sid, bulk, K_30)
        v_all = s_all[s_all > -999]
        v_k30 = s_k30[s_k30 > -999]

        vp = ax.violinplot(
            [v_all, v_k30],
            positions=[1, 2],
            showmedians=True,
            showmeans=True,
            showextrema=True,
        )
        vp["bodies"][0].set_facecolor("#95a5a6")
        vp["bodies"][0].set_alpha(0.7)
        vp["bodies"][1].set_facecolor("#e74c3c")
        vp["bodies"][1].set_alpha(0.7)

        for vals, pos, c in [(v_all, 1, "#2c3e50"), (v_k30, 2, "#c0392b")]:
            top3v = np.sort(vals)[::-1][:3]
            ax.scatter([pos] * 3, top3v, s=100, zorder=6, color=c, marker="*")
            ax.text(
                pos,
                ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else float(vals.min()) - 0.005,
                f"top3: {top3v.mean():.4f}\nσ={vals.std():.5f}",
                ha="center",
                fontsize=8,
                color=c,
            )

        ax.set_xticks([1, 2])
        ax.set_xticklabels(
            [
                f"All spots\n(100%)\nμ={v_all.mean():.4f}",
                f"K=30%\n({K_30})\nμ={v_k30.mean():.4f}",
            ],
            fontsize=9,
        )
        ax.set_ylabel("Tile-wise PCC")
        ax.set_title(f"{sid[-16:]}\nT={n_emb}", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/violin_tile_pcc_v2.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT_DIR}/violin_tile_pcc_v2.png")

    # ---------- Figures 2–3 per slide ----------
    for sid, bulk, wsi_path, hdf5_path, n_emb in selected:
        print(f"\nProcessing {sid}...")

        embs_n = np.load(f"{TCGA_EMB}/{sid}.npy").shape[0]
        coords, key_order = load_hdf5_coords(hdf5_path, sort_keys=SORT_HDF5_KEYS)

        if embs_n != len(coords):
            print(
                f"  ERROR: embedding rows ({embs_n}) != HDF5 tiles ({len(coords)}). "
                f"Fix embedding export or HDF5. SORT_HDF5_KEYS={SORT_HDF5_KEYS}",
                file=sys.stderr,
            )
            continue

        if SORT_HDF5_KEYS:
            print(
                f"  coords order: sorted HDF5 keys (first 3): {key_order[:3]} … "
                "Ensure train inference used the SAME key order."
            )

        s_all, _ = get_tile_scores_and_coords(sid, bulk, None)
        s_k30, _ = get_tile_scores_and_coords(sid, bulk, K_30)

        T = min(len(coords), len(s_all))
        coords = coords[:T]
        s_all = s_all[:T]
        s_k30 = s_k30[:T]

        img, scale_y, scale_x, fullH, fullW = load_wsi_lowres(wsi_path)
        lH, lW = img.shape[:2]
        # Level-0 footprint on slide (before 256 resize); scale to chosen WSI pyramid level.
        tile_px_lr_x = max(1, int(round(CROP_PX_LEVEL0 * scale_x)))
        tile_px_lr_y = max(1, int(round(CROP_PX_LEVEL0 * scale_y)))

        valid_mask = (s_all > -999) & (s_k30 > -999)
        vidx = np.where(valid_mask)[0]

        def top_bot3(scores: np.ndarray, vidx_local: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            vs = scores[vidx_local]
            order = np.argsort(vs)[::-1]
            top3 = vidx_local[order[:3]]
            bot3 = vidx_local[order[-3:]]
            return top3, bot3

        top3_all, bot3_all = top_bot3(s_all, vidx)
        top3_k30, bot3_k30 = top_bot3(s_k30, vidx)

        # -------- Figure 2: WSI --------
        fig2, axes2 = plt.subplots(1, 2, figsize=(22, 12))
        fig2.suptitle(f"Top-3 / Bot-3 Tile on WSI\n{sid}  (T={T})", fontsize=12)

        for ax, top3, bot3, scores, label, ct, cb in [
            (axes2[0], top3_all, bot3_all, s_all, "All spots (100%)", "#e74c3c", "#2980b9"),
            (axes2[1], top3_k30, bot3_k30, s_k30, f"K=30% ({K_30})", "#e74c3c", "#2980b9"),
        ]:
            ax.imshow(img, aspect="equal", origin="upper")

            for i in range(T):
                x_lr = int(coords[i, 0] * scale_x)
                y_lr = int(coords[i, 1] * scale_y)
                if y_lr < 0 or x_lr < 0 or y_lr >= lH or x_lr >= lW:
                    continue
                rect = mpatches.Rectangle(
                    (x_lr, y_lr),
                    tile_px_lr_x,
                    tile_px_lr_y,
                    linewidth=0.2,
                    edgecolor="white",
                    facecolor="none",
                    alpha=0.15,
                )
                ax.add_patch(rect)

            for rank, t_idx in enumerate(top3):
                x_lr = int(coords[t_idx, 0] * scale_x)
                y_lr = int(coords[t_idx, 1] * scale_y)
                if y_lr < 0 or y_lr >= lH or x_lr < 0 or x_lr >= lW:
                    print(
                        f"  WARNING: top3 tile {t_idx} out of bounds: y={y_lr}, x={x_lr}"
                    )
                    continue
                bw = tile_px_lr_x * 2
                bh = tile_px_lr_y * 2
                ax.add_patch(
                    mpatches.Rectangle(
                        (x_lr - bw // 4, y_lr - bh // 4),
                        bw,
                        bh,
                        linewidth=3,
                        edgecolor=ct,
                        facecolor=ct,
                        alpha=0.25,
                        zorder=5,
                    )
                )
                ax.add_patch(
                    mpatches.Rectangle(
                        (x_lr - bw // 4, y_lr - bh // 4),
                        bw,
                        bh,
                        linewidth=3,
                        edgecolor=ct,
                        facecolor="none",
                        zorder=6,
                    )
                )
                ax.text(
                    x_lr + tile_px_lr_x // 2,
                    y_lr - tile_px_lr_y // 2,
                    f"T{rank + 1}\n{scores[t_idx]:.4f}",
                    color=ct,
                    fontsize=9,
                    fontweight="bold",
                    ha="center",
                    va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85),
                    zorder=7,
                )

            for rank, t_idx in enumerate(bot3):
                x_lr = int(coords[t_idx, 0] * scale_x)
                y_lr = int(coords[t_idx, 1] * scale_y)
                if y_lr < 0 or y_lr >= lH or x_lr < 0 or x_lr >= lW:
                    continue
                bw = tile_px_lr_x * 2
                bh = tile_px_lr_y * 2
                ax.add_patch(
                    mpatches.Rectangle(
                        (x_lr - bw // 4, y_lr - bh // 4),
                        bw,
                        bh,
                        linewidth=3,
                        edgecolor=cb,
                        facecolor=cb,
                        alpha=0.20,
                        zorder=5,
                    )
                )
                ax.add_patch(
                    mpatches.Rectangle(
                        (x_lr - bw // 4, y_lr - bh // 4),
                        bw,
                        bh,
                        linewidth=3,
                        edgecolor=cb,
                        facecolor="none",
                        zorder=6,
                    )
                )
                ax.text(
                    x_lr + tile_px_lr_x // 2,
                    y_lr + tile_px_lr_y * 2,
                    f"B{rank + 1}\n{scores[t_idx]:.4f}",
                    color=cb,
                    fontsize=9,
                    fontweight="bold",
                    ha="center",
                    va="top",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85),
                    zorder=7,
                )

            top3_mean = float(np.mean(scores[top3])) if len(top3) else 0.0
            bot3_mean = float(np.mean(scores[bot3])) if len(bot3) else 0.0
            ax.set_title(
                f"{label}\nTop-3 PCC={top3_mean:.4f}  Bot-3 PCC={bot3_mean:.4f}",
                fontsize=10,
            )
            ax.axis("off")
            ax.legend(
                handles=[
                    Line2D([0], [0], color=ct, lw=3, label="Top-3 (high PCC)"),
                    Line2D([0], [0], color=cb, lw=3, label="Bot-3 (low PCC)"),
                ],
                fontsize=9,
                loc="lower right",
            )

        plt.tight_layout()
        out_path = f"{OUT_DIR}/wsi_top3_{sid}_v2.png"
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path}")

        # -------- Grid heatmap --------
        x_vals = coords[:T, 0]
        y_vals = coords[:T, 1]

        uy = np.unique(np.sort(y_vals))
        ux = np.unique(np.sort(x_vals))
        if len(uy) > 1:
            dy = np.diff(uy)
            stride_y = int(np.median(dy[dy > 0])) if np.any(dy > 0) else 512
        else:
            stride_y = 512
        if len(ux) > 1:
            dx = np.diff(ux)
            stride_x = int(np.median(dx[dx > 0])) if np.any(dx > 0) else 512
        else:
            stride_x = 512

        grid_row = ((y_vals - y_vals.min()) / max(stride_y, 1)).astype(int)
        grid_col = ((x_vals - x_vals.min()) / max(stride_x, 1)).astype(int)
        max_row = int(grid_row.max()) + 1
        max_col = int(grid_col.max()) + 1

        for k_label, scores, top3, bot3 in [
            ("All_100pct", s_all, top3_all, bot3_all),
            ("K30pct", s_k30, top3_k30, bot3_k30),
        ]:
            valid_scores = scores[scores > -999]
            vmin = np.percentile(valid_scores, 2)
            vmax = np.percentile(valid_scores, 98)

            fig3, axes3 = plt.subplots(1, 3, figsize=(20, 7))
            fig3.suptitle(f"{sid}\n{k_label}  T={T}", fontsize=11)

            ax = axes3[0]
            ax.hist(valid_scores, bins=60, color="#3498db", alpha=0.75, density=True)
            for t_idx in top3:
                if t_idx < T:
                    ax.axvline(scores[t_idx], color="#e74c3c", linewidth=2, alpha=0.9)
            for t_idx in bot3:
                if t_idx < T:
                    ax.axvline(
                        scores[t_idx],
                        color="#2980b9",
                        linewidth=2,
                        linestyle="--",
                        alpha=0.9,
                    )
            ax.set_xlabel("Tile PCC score")
            ax.set_ylabel("Density")
            ax.set_title("Score Distribution\nred=top3, blue=bot3")
            ax.legend(
                handles=[
                    Line2D([0], [0], color="#e74c3c", lw=2, label="top-3"),
                    Line2D([0], [0], color="#2980b9", lw=2, ls="--", label="bot-3"),
                ],
                fontsize=9,
            )
            ax.grid(alpha=0.3)

            ax = axes3[1]
            score_grid = np.full((max_row, max_col), np.nan)
            for i in range(T):
                if scores[i] > -999:
                    score_grid[grid_row[i], grid_col[i]] = scores[i]
            im = ax.imshow(
                score_grid,
                cmap="RdYlGn",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
                origin="upper",
            )
            plt.colorbar(im, ax=ax, fraction=0.03)

            for rank, t_idx in enumerate(top3):
                if t_idx < T:
                    ax.scatter(
                        grid_col[t_idx],
                        grid_row[t_idx],
                        marker="*",
                        s=300,
                        c="#e74c3c",
                        zorder=5,
                        edgecolors="white",
                        linewidths=0.5,
                    )
                    ax.text(
                        grid_col[t_idx] + 0.5,
                        grid_row[t_idx],
                        f"T{rank + 1}",
                        fontsize=8,
                        color="white",
                        fontweight="bold",
                    )
            for rank, t_idx in enumerate(bot3):
                if t_idx < T:
                    ax.scatter(
                        grid_col[t_idx],
                        grid_row[t_idx],
                        marker="X",
                        s=200,
                        c="#2980b9",
                        zorder=5,
                        edgecolors="white",
                        linewidths=0.5,
                    )

            ax.set_title("Score Heatmap\n★=Top-3, X=Bot-3")
            ax.set_xlabel("Grid col")
            ax.set_ylabel("Grid row")
            ax.legend(
                handles=[
                    Line2D(
                        [0],
                        [0],
                        marker="*",
                        color="w",
                        markerfacecolor="#e74c3c",
                        markersize=12,
                        label="Top-3",
                    ),
                    Line2D(
                        [0],
                        [0],
                        marker="X",
                        color="w",
                        markerfacecolor="#2980b9",
                        markersize=10,
                        label="Bot-3",
                    ),
                ],
                fontsize=9,
                loc="lower right",
            )

            ax = axes3[2]
            ax.scatter(grid_col, grid_row, s=5, c="#bdc3c7", alpha=0.5, zorder=1)
            cx = float(grid_col.mean())
            cy = float(grid_row.mean())
            ax.scatter([cx], [cy], s=100, c="black", marker="+", zorder=4, label="center")

            for rank, t_idx in enumerate(top3):
                if t_idx < T:
                    ax.scatter(
                        grid_col[t_idx],
                        grid_row[t_idx],
                        marker="*",
                        s=400,
                        c="#e74c3c",
                        zorder=5,
                        edgecolors="white",
                        linewidths=0.8,
                    )
                    ax.text(
                        grid_col[t_idx] + 0.5,
                        grid_row[t_idx],
                        f"T{rank + 1}\n({grid_col[t_idx]},{grid_row[t_idx]})",
                        fontsize=8,
                        color="#c0392b",
                        fontweight="bold",
                    )

            ax.set_title("Top-3 Spatial Position")
            ax.set_xlabel("Grid col")
            ax.set_ylabel("Grid row")
            ax.invert_yaxis()
            ax.legend(fontsize=9)
            ax.grid(alpha=0.2)

            plt.tight_layout()
            out_path2 = f"{OUT_DIR}/grid_heatmap_{sid}_{k_label}.png"
            plt.savefig(out_path2, dpi=130, bbox_inches="tight")
            plt.close()
            print(f"  Saved: {out_path2}")

    print("\nAll done!")


if __name__ == "__main__":
    main()
