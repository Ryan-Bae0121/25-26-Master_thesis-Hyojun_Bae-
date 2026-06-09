# 65μm 모델용으로 경로 오버라이드해서 실행

import os, glob, h5py, numpy as np
from pathlib import Path
from PIL import Image
import torch, torch.nn.functional as F
import open_clip, pandas as pd
from tqdm import tqdm

# ── 65μm 전용 설정 ────────────────────────────────────────
PATCH_DIR  = "/project_antwerp/TCGA-HNSC/TCGA_patch"
CKPT_PATH  = "/project_antwerp/hbae/Loki_output/65um_finetune_fold_03/finetune_65um_fold_03/checkpoints/epoch_latest.pt"
REF_FILE   = "/project_antwerp/hbae/ref_file.csv"
OUTPUT_DIR = "/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/TCGA_embeddings_65um/fold_03"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# 슬라이드 목록
ref_df = pd.read_csv(REF_FILE, index_col=0)
ref_df["slide_id"] = ref_df["wsi_file_name"].apply(lambda x: x.split(".")[0])
patch_dirs = {p.name: p for p in Path(PATCH_DIR).iterdir() if p.is_dir()}
matched = [(row["slide_id"], patch_dirs[row["slide_id"]])
           for _, row in ref_df.iterrows()
           if row["slide_id"] in patch_dirs]
print(f"Matched slides: {len(matched)}")

# 모델 로드
print(f"Loading checkpoint: {CKPT_PATH}")
model, _, preprocess = open_clip.create_model_and_transforms("coca_ViT-L-14")
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
sd   = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt.get("model", ckpt)))
model.load_state_dict(sd, strict=False)
model = model.to(device).eval()
print("Model loaded!")

@torch.no_grad()
def extract_embs(pil_images, batch_size=256):
    all_embs = []
    for i in range(0, len(pil_images), batch_size):
        batch = pil_images[i:i+batch_size]
        t = torch.stack([preprocess(img) for img in batch]).to(device)
        e = F.normalize(model.encode_image(t), dim=-1)
        all_embs.append(e.cpu())
    return torch.cat(all_embs, dim=0).numpy()

# 추출
for sid, patch_path in tqdm(matched, desc="Extracting"):
    emb_path    = f"{OUTPUT_DIR}/{sid}.npy"
    coords_path = f"{OUTPUT_DIR}/{sid}_coords.npy"
    if os.path.exists(emb_path) and os.path.exists(coords_path):
        continue

    hdf5_files = list(patch_path.glob("*.hdf5"))
    if not hdf5_files:
        continue

    try:
        imgs, coords = [], []
        with h5py.File(hdf5_files[0], "r") as f:
            for k in f.keys():
                imgs.append(Image.fromarray(f[k][:]))
                y, x = map(int, k.split("_"))
                coords.append([y, x])
    except Exception as e:
        print(f"  SKIP {sid}: {e}")
        continue

    if not imgs:
        continue

    embs = extract_embs(imgs)
    np.save(emb_path,    embs)
    np.save(coords_path, np.array(coords))

print(f"\nDone! Saved to: {OUTPUT_DIR}")
print(f"Total files: {len(glob.glob(OUTPUT_DIR+'/*.npy'))}")
