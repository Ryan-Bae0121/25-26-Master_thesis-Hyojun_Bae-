
import h5py
import numpy as np
import pandas as pd
from pathlib import Path

# Gene names from h5
with h5py.File('/project_antwerp/hbae/data/Open_ST/openst_patches_agg_mc10.h5', 'r') as f:
    gene_names = [g.decode() if isinstance(g, bytes) else g 
                  for g in f.attrs['gene_names']]

out_dir = Path('/project_antwerp/hbae/figures/Appendix/openst')
out_dir.mkdir(parents=True, exist_ok=True)

# S1: All 1,946 shared genes
with open(out_dir / 'OpenST_shared_1946_genes.txt', 'w') as f:
    f.write('\n'.join(gene_names))
print(f'S1: {len(gene_names)} genes')

# S2: HEG top-300 (from agg_v2 fold_01)
pcc_heg = np.load('/project_antwerp/hbae/Loki_output/Open_ST/openst_validation_agg_v2/fold_01/openst_genewise_pcc.npy')
# fold_01 genewise_pcc.npy shape 확인 필요 - gene 이름과 매핑되는지
# 만약 CSV라면 아래로 대체
heg_df = pd.read_csv('/project_antwerp/hbae/Loki_output/Open_ST/openst_validation_agg_v2/fold_01/openst_genewise_pcc.csv') \
         if Path('/project_antwerp/hbae/Loki_output/Open_ST/openst_validation_agg_v2/fold_01/openst_genewise_pcc.csv').exists() \
         else None
if heg_df is not None:
    heg_genes = heg_df['gene'].tolist()
    with open(out_dir / 'OpenST_HEG300_genes.txt', 'w') as f:
        f.write('\n'.join(heg_genes))
    print(f'S2: {len(heg_genes)} genes')

# S3: HVG 300
hvg_df = pd.read_csv('/project_antwerp/hbae/Loki_output/Open_ST/openst_validation_hvg300/openst_genewise_pcc.csv')
with open(out_dir / 'OpenST_HVG300_genes.txt', 'w') as f:
    f.write('\n'.join(hvg_df['gene'].tolist()))
print(f'S3: {len(hvg_df)} genes')

# S4: Oracle 300
oracle_df = pd.read_csv('/project_antwerp/hbae/Loki_output/Open_ST/openst_validation_TopPCC300/fold_01/openst_genewise_pcc_toppcc.csv')
with open(out_dir / 'OpenST_oracle300_genes.txt', 'w') as f:
    f.write('\n'.join(oracle_df['gene'].tolist()))
print(f'S4: {len(oracle_df)} genes')
