import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import os

# baseline은 ksweep_331 폴더에 저장됨
KSWEEP_DIR = "/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/144px_finetuned_ksweep_331"
OUTPUT_DIR = "/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/tile_selection_331"

baseline_preds = np.load(os.path.join(KSWEEP_DIR, "baseline_pred.npy"))
bulk_arr       = np.load(os.path.join(OUTPUT_DIR, "bulk_arr.npy"))

pccs = []
for i in range(baseline_preds.shape[1]):
    p, b = baseline_preds[:, i], bulk_arr[:, i]
    if p.std() < 1e-8 or b.std() < 1e-8:
        continue
    r, _ = pearsonr(p, b)
    pccs.append(r)
pccs = np.array(pccs)
print(f"All tiles | mean={pccs.mean():.4f} | median={np.median(pccs):.4f}")