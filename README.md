# HNSCC Spatial Transcriptomics — Thesis Code (multi-server)

Combined code repository for an HNSCC spatial transcriptomics thesis. Work was
done across multiple servers; each server's code lives in its own top-level
folder to avoid path/import conflicts.

> **Code only.** Raw/processed data, model checkpoints, virtual environments,
> and credentials are excluded via the root `.gitignore` and must be obtained
> separately.

## Layout

```
.
├── agro/        # code from the `agro.bioit.labnet` server
│   └── README.md   (detailed: WSI tiling, ST preprocessing, LOKI, figures)
└── <other>/     # code from the other server (added from that machine)
```

Each server folder is self-contained (its own scripts, modules, and README).
See the per-folder `README.md` for details, data sources, and usage.

## Adding the other server's code

On the other server:

```bash
git clone <repo-url>
cd <repo>
mkdir <other-server-name>
# copy that server's project code into <other-server-name>/
git add <other-server-name>
git commit -m "Add <other-server-name> server code"
git push
```

Then on this (agro) server, run `git pull` to sync.
