
import pandas as pd
from pathlib import Path

out_dir = Path('/project_antwerp/hbae/figures/Appendix/tcga')
out_dir.mkdir(parents=True, exist_ok=True)

# TCGA shared 1,968 genes - fold_01 기준
# 어느 폴더에 gene list가 있는지 확인
import glob
files = glob.glob('/project_antwerp/hbae/Loki_output/TCGA*/fold_01/*.csv')
for f in files:
    print(f)
