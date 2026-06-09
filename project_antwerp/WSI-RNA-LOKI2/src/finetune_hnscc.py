import subprocess
import os
from pathlib import Path
import time
# 경로들 (환경 변수로 설정 가능, 기본값 제공)
OPENCLIP_ROOT = Path(os.getenv("OPENCLIP_ROOT", "/project_antwerp/hbae/open_clip2"))  # open_clip 레포 위치
PRETRAINED = Path(os.getenv("PRETRAINED", "/project_antwerp/assets/loki_ckpts/checkpoint.pt"))  # omiclip ckpt
CSV_PATH = Path(os.getenv("CSV_PATH", "/project_antwerp/hbae/data/HVG_finetune_meta_gpulab.csv"))

# 모델 설정 (환경 변수로 오버라이드 가능)
model_name = os.getenv("MODEL_NAME", "coca_ViT-L-14")
name = os.getenv("EXPERIMENT_NAME", f"finetune_HVG_hnscc_{int(time.time())}")

# 하이퍼파라미터 (환경 변수로 오버라이드 가능)
epochs = os.getenv("EPOCHS", "5")
batch_size = os.getenv("BATCH_SIZE", "64")
lr = os.getenv("LR", "5e-6")
wd = os.getenv("WD", "0.1")
warmup = os.getenv("WARMUP", "10")
workers = os.getenv("WORKERS", "16")

train_command = [
    "python", "-u", "-m", "open_clip_train.main",
    "--name", name,
    "--train-data", str(CSV_PATH),
    "--csv-img-key", "img_path",
    "--csv-caption-key", "label",
    "--csv-separator", ",",

    "--model", model_name,
    "--pretrained", str(PRETRAINED),
    "--device", "cuda",  # GPU 모드 사용

    "--epochs", epochs,
    "--batch-size", batch_size,
    "--lr", lr,
    "--wd", wd,
    "--warmup", warmup,
    "--workers", workers,

    "--lock-text-freeze-layer-norm",
    "--lock-image-freeze-bn-stats",
    "--coca-caption-loss-weight", "0",
    "--coca-contrastive-loss-weight", "1",

    "--save-frequency", "1",
    "--val-frequency", "10",
    "--report-to", "wandb",  # WandB 활성화
    "--debug",

    "--aug-cfg",
    "color_jitter=(0.32, 0.32, 0.32, 0.08)",
    "color_jitter_prob=0.5",
    "gray_scale_prob=0",
    "--save-most-recent"

]

# ❗ 핵심: cwd를 open_clip 루트로 설정하고 PYTHONPATH에 src 추가
env = os.environ.copy()
src_path = str(OPENCLIP_ROOT / "src")
if "PYTHONPATH" in env:
    env["PYTHONPATH"] = f"{src_path}:{env['PYTHONPATH']}"
else:
    env["PYTHONPATH"] = src_path

# WandB API 키 설정
# 환경 변수에 API 키가 없으면 기본값 사용
if "WANDB_API_KEY" not in env:
    env["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY", "a8d69bdb4fd712911b58b798d2045e86cd34a4d0")
    print("✓ WandB API 키가 환경 변수에 설정되었습니다.")


# 경로 확인
print(f"OpenCLIP Root: {OPENCLIP_ROOT}")
print(f"Pretrained: {PRETRAINED}")
print(f"CSV Path: {CSV_PATH}")
print(f"Model: {model_name}")
print(f"Experiment Name: {name}")

# 출력을 실시간으로 볼 수 있도록 설정 (디버그 모드)
subprocess.run(train_command, cwd=str(OPENCLIP_ROOT), env=env)

