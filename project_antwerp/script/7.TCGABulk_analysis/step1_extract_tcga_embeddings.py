"""
Script 1: TCGA Tile Embedding 추출 및 저장 (--fold 인자 지원)
"""

import os
import glob
import h5py
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
import open_clip
import pandas as pd
from tqdm import tqdm

# ─── 인자 파싱 ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=str, default=None,
                    help="특정 fold만 실행 (예: fold_01). 없으면 전체 실행.")
args = parser.parse_args()

# ─── 경로 설정 ────────────────────────────────────────────────────────────────
PATCH_DIR    = "/project_antwerp/TCGA-HNSC/TCGA_patch"
FINETUNE_DIR = "/project_antwerp/hbae/Loki_output/0317_10epoch_finetune_10fold_runs_hvg_"
REF_FILE     = "/project_antwerp/hbae/ref_file.csv"
OUTPUT_DIR   = "/project_antwerp/hbae/data"

ref_df = pd.read_csv(REF_FILE, index_col=0)
ref_df["slide_id"] = ref_df["wsi_file_name"].apply(lambda x: x.split(".")[0])
patch_dirs = {p.name: p for p in Path(PATCH_DIR).iterdir() if p.is_dir()}
matched_sids = [
    (row["slide_id"], patch_dirs[row["slide_id"]])
    for _, row in ref_df.iterrows()
    if row["slide_id"] in patch_dirs
]
print(f"Matched slides: {len(matched_sids)}")

ALL_FOLDS = [f"fold_{i:02d}" for i in range(1, 11)]
FOLDS     = [args.fold] if args.fold else ALL_FOLDS
print(f"Target folds: {FOLDS}")

# ─── GPU 설정 ─────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def find_checkpoint(fold_name):
    pattern = os.path.join(FINETUNE_DIR, fold_name, "finetune_hvg_*", "checkpoints", "epoch_latest.pt")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No checkpoint: {pattern}")
    return matches[0]

def load_model(ckpt_path):
    model, _, preprocess = open_clip.create_model_and_transforms("coca_ViT-L-14")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt))
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    return model, preprocess

@torch.no_grad()
def extract_embeddings(pil_images, model, preprocess, batch_size=256):
    all_embs = []
    for i in range(0, len(pil_images), batch_size):
        batch   = pil_images[i:i+batch_size]
        tensors = torch.stack([preprocess(img) for img in batch]).to(device)
        embs    = model.encode_image(tensors)
        embs    = F.normalize(embs, dim=-1)
        all_embs.append(embs.cpu())
    return torch.cat(all_embs, dim=0).numpy()

# ─── 메인 ─────────────────────────────────────────────────────────────────────
for fold in FOLDS:
    fold_out = os.path.join(OUTPUT_DIR, fold)
    os.makedirs(fold_out, exist_ok=True)

    existing = len(glob.glob(os.path.join(fold_out, "*.npy")))
    expected = len(matched_sids) * 2
    if existing >= expected:
        print(f"\n[{fold}] Already done ({existing} files). Skipping.")
        continue

    print(f"\n{'='*50}")
    print(f"  Fold: {fold}")
    print(f"{'='*50}")

    try:
        ckpt_path = find_checkpoint(fold)
        print(f"  Checkpoint: {ckpt_path}")
    except FileNotFoundError as e:
        print(f"  [SKIP] {e}")
        continue

    model, preprocess = load_model(ckpt_path)
    print(f"  Model loaded.")

    for sid, patch_path in tqdm(matched_sids, desc=f"  {fold}"):
        emb_path    = os.path.join(fold_out, f"{sid}.npy")
        coords_path = os.path.join(fold_out, f"{sid}_coords.npy")

        if os.path.exists(emb_path) and os.path.exists(coords_path):
            continue

        hdf5_files = list(patch_path.glob("*.hdf5"))
        if not hdf5_files:
            continue

        try:
            pil_images, coords = [], []
            with h5py.File(hdf5_files[0], "r") as f:
                for k in f.keys():
                    pil_images.append(Image.fromarray(f[k][:]))
                    y_px, x_px = map(int, k.split("_"))
                    coords.append([y_px, x_px])
        except Exception as e:
            tqdm.write(f"    [SKIP] {sid}: {e}")
            continue

        if not pil_images:
            continue

        embs = extract_embeddings(pil_images, model, preprocess, batch_size=256)
        np.save(emb_path,    embs)
        np.save(coords_path, np.array(coords))

    del model
    torch.cuda.empty_cache()
    print(f"  Done. Memory freed.")

print("\nAll embeddings extracted!")
print(f"Saved to: {OUTPUT_DIR}")
