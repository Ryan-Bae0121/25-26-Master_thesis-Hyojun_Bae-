#!/usr/bin/env bash
# Download and verify official Loki demo checkpoint
# Based on: https://guangyuwanglab2021.github.io/Loki/notebooks/basic_usage.html

set -euo pipefail

ZIP=./loki_demo_data.zip
OUT=./loki_demo_data
FILE_ID=1aPK1nItsOEPxTihUAKMig-vLY-DMMIce

echo "=================================================================="
echo "Loki Demo Checkpoint Downloader & Verifier"
echo "=================================================================="

# 0) Ensure gdown is installed
echo "[INFO] Checking gdown..."
python - <<'PY'
import sys, subprocess
try:
    import gdown  # noqa
    print("[INFO] ✅ gdown is installed")
except Exception:
    print("[INFO] Installing gdown...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gdown"])
    print("[INFO] ✅ gdown installed")
PY

# 1) Download if needed
if [ -f "$ZIP" ] && [ -s "$ZIP" ]; then
    SIZE=$(stat -c%s "$ZIP" 2>/dev/null || stat -f%z "$ZIP" 2>/dev/null || echo 0)
    if [ "$SIZE" -gt 10000000 ]; then
        echo "[INFO] ✅ Demo zip already exists: $ZIP ($(numfmt --to=iec $SIZE 2>/dev/null || echo ${SIZE} bytes))"
    else
        echo "[WARN] Existing zip is too small, re-downloading..."
        rm -f "$ZIP"
    fi
fi

if [ ! -f "$ZIP" ]; then
    echo "[INFO] Downloading demo zip from Google Drive..."
    echo "[INFO] File ID: $FILE_ID"
    echo "[INFO] This may take several minutes (6.85 GB)..."
    
    python - <<PY
import gdown
url = "https://drive.google.com/uc?id=${FILE_ID}"
gdown.download(url, "${ZIP}", quiet=False)
print("[INFO] ✅ Download complete")
PY
    
    if [ ! -f "$ZIP" ]; then
        echo "[ERROR] Download failed!"
        exit 1
    fi
fi

echo "[INFO] Zip file: $ZIP"
ls -lh "$ZIP"

# 2) Unzip (idempotent)
echo "[INFO] Extracting demo data..."
mkdir -p "$OUT"

python - <<'PY'
import zipfile
import os

zip_path = "loki_demo_data.zip"
out_dir = "loki_demo_data"

print(f"[INFO] Extracting {zip_path} to {out_dir}/...")

with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(out_dir)

print("[INFO] ✅ Extraction complete")
PY

# 3) Verify checkpoint location
CKPT="./loki_demo_data/data/basic_usage/checkpoint.pt"

echo "[INFO] Verifying checkpoint location..."
echo "[INFO] Expected: $CKPT"

if [ -f "$CKPT" ]; then
    echo "[OK] ✅ Demo checkpoint found!"
    echo "[INFO] Checkpoint: $CKPT"
    
    # Save absolute path
    realpath "$CKPT" > .demo_ckpt_path.txt
    
    # Show file size
    ls -lh "$CKPT"
    
    echo ""
    echo "=================================================================="
    echo "✅ Demo checkpoint ready for use"
    echo "=================================================================="
    echo "Path saved to: .demo_ckpt_path.txt"
    
    exit 0
else
    echo "[ERROR] ❌ checkpoint.pt not found at expected location!"
    echo "[ERROR] Expected: $CKPT"
    echo ""
    echo "[HINT] Listing zip contents to debug..."
    
    python - <<'PY'
import zipfile

with zipfile.ZipFile("loki_demo_data.zip", "r") as z:
    names = z.namelist()
    
    # Show first 40 entries
    print("\nFirst 40 entries in zip:")
    for n in names[:40]:
        print(f"  - {n}")
    
    # Search for checkpoint files
    ckpt_files = [n for n in names if 'checkpoint' in n.lower() and n.endswith('.pt')]
    if ckpt_files:
        print(f"\nFound {len(ckpt_files)} checkpoint files:")
        for f in ckpt_files:
            print(f"  - {f}")
    else:
        print("\n⚠️  No .pt checkpoint files found in zip")
    
    print(f"\nTotal entries: {len(names)}")
PY
    
    echo ""
    echo "[ERROR] Cannot proceed without checkpoint"
    exit 2
fi

