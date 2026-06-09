import subprocess
import os
from pathlib import Path
import pandas as pd

# =========================================
# 0. 기본 경로 설정
# =========================================

# 기본적으로 open_clip repo 위치
OPENCLIP_ROOT = Path(os.getenv("OPENCLIP_ROOT", "/project_antwerp/hbae/open_clip2"))

# Loki/OmicCLIP pretrained checkpoint
PRETRAINED = Path(os.getenv("PRETRAINED", "/project_antwerp/assets/loki_ckpts/checkpoint.pt"))

# 새 PredEx용 train_df.csv
TRAIN_DF = Path(os.getenv(
    "TRAIN_DF",
    "/project_antwerp/hbae/data/train_df_fixed.csv"
))

OUTPUT_DIR = Path(os.getenv(
    "OUTPUT_DIR",
    "/project_antwerp/hbae/experiments/loki_predex_finetune"
))


# =========================================
# 1. Train / Validation split 생성
# =========================================

train_csv = OUTPUT_DIR / "train_split.csv"
val_csv   = OUTPUT_DIR / "val_split.csv"

if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("🔍 Reading full train_df.csv ...")
df = pd.read_csv(TRAIN_DF)
print(f"Total spots: {len(df):,}")

# 🔧 1) 이미지 경로 컬럼 alias 만들기
# 기존: img_path  -> open_clip 쪽이 기대: filepath
if "filepath" not in df.columns:
    if "img_path" in df.columns:
        df["filepath"] = df["img_path"]
        print("✅ Added 'filepath' column from 'img_path'")
    else:
        raise ValueError("CSV에 'img_path' 컬럼이 없습니다. 이미지 경로 컬럼을 확인하세요.")

# 🔧 2) 캡션 컬럼 alias 만들기 (나중 에러 방지용)
# 기존: label -> open_clip 기본은 보통 'caption' 또는 'title'
if "caption" not in df.columns:
    if "label" in df.columns:
        df["caption"] = df["label"]
        print("✅ Added 'caption' column from 'label'")
    else:
        raise ValueError("CSV에 'label' 컬럼이 없습니다. 캡션 컬럼을 확인하세요.")
print("🔀 Creating 90/10 train/val split...")
val_df = df.sample(frac=0.10, random_state=42)
train_df = df.drop(val_df.index)

train_df.to_csv(train_csv, index=False)
val_df.to_csv(val_csv, index=False)



# =========================================
# 2. Fine-tuning 설정
# =========================================

model_name = os.getenv("MODEL_NAME", "coca_ViT-L-14")
experiment_name = os.getenv("EXPERIMENT_NAME", "finetune_hnscc_predex")

epochs     = os.getenv("EPOCHS", "5")
batch_size = os.getenv("BATCH_SIZE", "64")
lr         = os.getenv("LR", "5e-6")
wd         = os.getenv("WD", "0.1")
warmup     = os.getenv("WARMUP", "10")
workers    = os.getenv("WORKERS", "16")

train_command = [
    "python", "-u", "-m", "open_clip_train.main",

    "--name", experiment_name,

    # Train & Validation data
    "--train-data", str(train_csv),
    "--val-data", str(val_csv),
    "--csv-img-key", "filepath",
    "--csv-caption-key", "caption",
    "--csv-separator", ",",



    # Model
    "--model", model_name,
    "--pretrained", str(PRETRAINED),
    "--device", "cuda",

    # Hyperparameters
    "--epochs", epochs,
    "--batch-size", batch_size,
    "--lr", lr,
    "--wd", wd,
    "--warmup", warmup,
    "--workers", workers,

    # Contrastive-only training (COCA)
    "--lock-text-freeze-layer-norm",
    "--lock-image-freeze-bn-stats",
    "--coca-caption-loss-weight", "0",
    "--coca-contrastive-loss-weight", "1",

    # Validation & best checkpoint
    "--val-frequency", "1",
    "--save-frequency", "1",

    # Logging
    "--report-to", "wandb",
    "--log-every-n-steps", "50",

    # Data augmentation
    "--aug-cfg",
    "color_jitter=(0.32, 0.32, 0.32, 0.08)",
    "color_jitter_prob=0.5",
    "gray_scale_prob=0",
]

# =========================================
# 3. Python PATH 설정
# =========================================

env = os.environ.copy()
src_path = str(OPENCLIP_ROOT / "src")

if "PYTHONPATH" in env:
    env["PYTHONPATH"] = f"{src_path}:{env['PYTHONPATH']}"
else:
    env["PYTHONPATH"] = src_path

# WandB API key (optional override)
if "WANDB_API_KEY" not in env:
    env["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY", "YOUR_WANDB_KEY_HERE")
    print("✓ WandB API key set automatically")

# =========================================
# 4. 정보 출력
# =========================================

print("\n==================== CONFIG ====================")
print(f"OPENCLIP_ROOT    : {OPENCLIP_ROOT}")
print(f"PRETRAINED       : {PRETRAINED}")
print(f"TRAIN CSV        : {train_csv}")
print(f"VAL CSV          : {val_csv}")
print(f"Experiment name  : {experiment_name}")
print(f"Epochs           : {epochs}")
print(f"Batch size       : {batch_size}")
print("================================================\n")

# =========================================
# 5. 파인튜닝 실행
# =========================================

print("🚀 Starting Loki PredEx fine-tuning ...")
subprocess.run(train_command, cwd=str(OPENCLIP_ROOT), env=env)
print("🎉 Training finished!")
