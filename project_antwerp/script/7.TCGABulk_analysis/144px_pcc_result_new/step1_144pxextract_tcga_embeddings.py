"""
TCGA Tile Embedding 추출 (144px tiles.h5 버전, fast preprocess + bs=512)
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
import torchvision.transforms as T
import open_clip
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=str, default=None)
args = parser.parse_args()

PATCH_DIR    = "/project_antwerp/hbae/data/TCGA_HNSC_tiles_144px_h5"
FINETUNE_DIR = "/project_antwerp/hbae/Loki_output/0317_10epoch_finetune_10fold_runs_hvg_"
REF_FILE     = "/project_antwerp/hbae/ref_file.csv"
OUTPUT_DIR   = "/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/TCGA_embeddings_144px"

os.makedirs(OUTPUT_DIR, exist_ok=True)

ref_df = pd.read_csv(REF_FILE, index_col=0)
ref_df["slide_id"] = ref_df["wsi_file_name"] + ".svs"
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# GPU 배치 전처리 (PIL loop 대신)
fast_transform = T.Compose([
    T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)),
])

def find_checkpoint(fold_name):
    pattern = os.path.join(FINETUNE_DIR, fold_name, "finetune_hvg_*", "checkpoints", "epoch_latest.pt")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No checkpoint: {pattern}")
    return matches[0]

def load_model(ckpt_path):
    model, _, _ = open_clip.create_model_and_transforms("coca_ViT-L-14")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt))
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    return model

@torch.no_grad()
def extract_embeddings(images_np, model, batch_size=512):
    """images_np: (N, 144, 144, 3) uint8 — GPU 배치 전처리"""
    all_embs = []
    for i in range(0, len(images_np), batch_size):
        chunk = images_np[i:i+batch_size]
        arr = torch.from_numpy(chunk).permute(0,3,1,2).float().div(255).to(device)
        tensors = fast_transform(arr)
        embs = model.encode_image(tensors)
        embs = F.normalize(embs, dim=-1)
        all_embs.append(embs.cpu())
    return torch.cat(all_embs, dim=0).numpy()

for fold in FOLDS:
    fold_out = os.path.join(OUTPUT_DIR, fold)
    os.makedirs(fold_out, exist_ok=True)

    existing = len(glob.glob(os.path.join(fold_out, "*.npy")))
    expected = len(matched_sids) * 2
    if existing >= expected:
        print(f"\n[{fold}] Already done ({existing} files). Skipping.")
        continue

    print(f"\n{'='*50}\n  Fold: {fold}\n{'='*50}")

    try:
        ckpt_path = find_checkpoint(fold)
        print(f"  Checkpoint: {ckpt_path}")
    except FileNotFoundError as e:
        print(f"  [SKIP] {e}")
        continue

    model = load_model(ckpt_path)
    print(f"  Model loaded.")

    for sid, patch_path in tqdm(matched_sids, desc=f"  {fold}"):
        emb_path    = os.path.join(fold_out, f"{sid}.npy")
        coords_path = os.path.join(fold_out, f"{sid}_coords.npy")

        if os.path.exists(emb_path) and os.path.exists(coords_path):
            continue

        h5_path = patch_path / "tiles.h5"
        if not h5_path.exists():
            tqdm.write(f"    [SKIP] {sid}: tiles.h5 not found")
            continue

        try:
            with h5py.File(h5_path, "r") as f:
                images_np = f["images"][:]  # (N, 144, 144, 3)
                coords_np = f["coords"][:]  # (N, 2)
        except Exception as e:
            tqdm.write(f"    [SKIP] {sid}: {e}")
            continue

        if len(images_np) == 0:
            continue

        embs = extract_embeddings(images_np, model, batch_size=512)
        np.save(emb_path,    embs)
        np.save(coords_path, coords_np)

    del model
    torch.cuda.empty_cache()
    print(f"  Done. Memory freed.")

print(f"\nAll embeddings extracted!\nSaved to: {OUTPUT_DIR}")