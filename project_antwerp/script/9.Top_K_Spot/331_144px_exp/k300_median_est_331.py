import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import os

OUTPUT_DIR = "/project_antwerp/hbae/Loki_output/TCGA_bulk_prediction/tile_selection_331"

sweep_df = pd.read_csv(os.path.join(OUTPUT_DIR, "tile_selection_sweep.csv"))
pcc_rows = sweep_df[sweep_df["method"] == "tile_pcc"].copy()
pcc_rows["k_tiles"] = pcc_rows["k_tiles"].astype(float)

print("Existing mean values:")
print(pcc_rows[["k_tiles", "gene_pcc", "slide_pcc"]].to_string())

# K=300 선형 보간
k200 = pcc_rows[pcc_rows["k_tiles"] == 200.0]["gene_pcc"].values[0]
k500 = pcc_rows[pcc_rows["k_tiles"] == 500.0]["gene_pcc"].values[0]
k300_est = k200 + (k500 - k200) * (300 - 200) / (500 - 200)
print(f"\nK=300 estimated: {k300_est:.4f}")

# top500 predictions로 median 추정
top500_preds = np.load(os.path.join(OUTPUT_DIR, "top500_preds.npy"))
bulk_arr     = np.load(os.path.join(OUTPUT_DIR, "bulk_arr.npy"))

pccs = []
for i in range(top500_preds.shape[1]):
    p, b = top500_preds[:, i], bulk_arr[:, i]
    if p.std() < 1e-8 or b.std() < 1e-8:
        continue
    r, _ = pearsonr(p, b)
    pccs.append(r)
pccs = np.array(pccs)
ratio = np.median(pccs) / pccs.mean()
print(f"K=500 | mean={pccs.mean():.4f} | median={np.median(pccs):.4f}")
print(f"median/mean ratio: {ratio:.4f}")

print(f"\n{'K':>6} | {'mean':>8} | {'median_est':>12} | {'slide_pcc':>10}")
print(f"  {'-'*45}")
for _, row in pcc_rows.sort_values("k_tiles").iterrows():
    k    = int(row["k_tiles"])
    mean = row["gene_pcc"]
    med  = mean * ratio
    spcc = row["slide_pcc"]
    print(f"{k:>6} | {mean:>8.4f} | {med:>12.4f} | {spcc:>10.4f}")
print(f"{'300':>6} | {k300_est:>8.4f} | {k300_est*ratio:>12.4f} | {'~0.752':>10}")