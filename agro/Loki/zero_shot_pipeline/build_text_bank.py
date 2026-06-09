#!/usr/bin/env python3
"""
Build text bank from Visium AnnData for zero-shot retrieval.
Generates spot text embeddings and expression matrix.
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add Loki src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
try:
    import loki.utils
except ImportError:
    # Try alternative import
    import sys
    loki_src = str(Path(__file__).parent.parent / "src")
    if loki_src not in sys.path:
        sys.path.insert(0, loki_src)
    import loki.utils


def preprocess_gene_names(genes):
    """Uppercase and strip Ensembl version"""
    processed = []
    for g in genes:
        g = str(g).upper()
        # Strip .1, .2, etc from Ensembl IDs
        if '.' in g:
            g = g.split('.')[0]
        processed.append(g)
    return processed


def load_housekeeping(path):
    """Load housekeeping genes to exclude"""
    if path is None or not Path(path).exists():
        return set()
    with open(path, 'r') as f:
        genes = {line.strip().upper() for line in f if line.strip()}
    return genes


def build_spot_texts(adata, topn=50, housekeeping=None):
    """Build text string for each spot from top-N genes"""
    print(f"🔤 Building spot texts (top-{topn} genes per spot)...")
    
    if housekeeping is None:
        housekeeping = set()
    
    # Get expression matrix (spots × genes)
    if hasattr(adata.X, 'toarray'):
        expr = adata.X.toarray()
    else:
        expr = adata.X
    
    spot_texts = []
    
    for i in tqdm(range(expr.shape[0]), desc="Processing spots"):
        spot_expr = expr[i, :]
        
        # Get top-N gene indices
        top_indices = np.argsort(spot_expr)[::-1]
        
        # Filter out housekeeping and get top-N
        top_genes = []
        for idx in top_indices:
            gene = adata.var_names[idx]
            if gene not in housekeeping:
                top_genes.append(gene)
                if len(top_genes) >= topn:
                    break
        
        # Create text string
        text = " ".join(top_genes)
        spot_texts.append(text)
    
    print(f"   ✅ Generated {len(spot_texts)} spot texts")
    return spot_texts


def encode_spot_texts(model, tokenizer, texts, device, batch_size=256):
    """Encode spot texts to embeddings"""
    print(f"🧬 Encoding {len(texts)} spot texts...")
    
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding batches"):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize
        text_inputs = tokenizer(batch_texts)
        
        # Move text inputs to device
        if isinstance(text_inputs, dict):
            text_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                          for k, v in text_inputs.items()}
        elif isinstance(text_inputs, torch.Tensor):
            text_inputs = text_inputs.to(device)
        
        # Encode
        with torch.no_grad():
            embeddings = model.encode_text(text_inputs)
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        all_embeddings.append(embeddings.cpu().numpy())
    
    embeddings = np.vstack(all_embeddings)
    print(f"   ✅ Text embeddings shape: {embeddings.shape}")
    
    return embeddings


def main():
    parser = argparse.ArgumentParser(description='Build text bank from Visium AnnData')
    parser.add_argument('--bank_h5ad', type=str, required=True,
                       help='Path to Visium bank AnnData (.h5ad)')
    parser.add_argument('--hf_ckpt', type=str, default=None,
                       help='Path to OmiCLIP checkpoint.pt (not needed if --use_demo)')
    parser.add_argument('--out_dir', type=str, required=True,
                       help='Output directory for bank files')
    parser.add_argument('--topn', type=int, default=50,
                       help='Top-N genes per spot for text (default: 50)')
    parser.add_argument('--min_counts', type=int, default=0,
                       help='Minimum counts per spot (filter)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--housekeeping', type=str, default=None,
                       help='Path to housekeeping genes file (optional)')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for encoding')
    parser.add_argument('--use_demo', action='store_true',
                       help='Use lightweight demo checkpoint (no HF weights needed)')
    
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Building Text Bank for Zero-Shot Retrieval")
    print("=" * 70)
    
    start_time = time.time()
    
    # Load AnnData
    print(f"\n📂 Loading AnnData: {args.bank_h5ad}")
    adata = ad.read_h5ad(args.bank_h5ad)
    print(f"   ✅ Loaded: {adata.shape[0]} spots × {adata.shape[1]} genes")
    
    # Filter by min counts
    if args.min_counts > 0:
        sc.pp.filter_cells(adata, min_counts=args.min_counts)
        print(f"   ✅ After filtering (min_counts={args.min_counts}): {adata.shape[0]} spots")
    
    # Preprocess gene names
    print("\n🧬 Preprocessing gene names...")
    adata.var_names = preprocess_gene_names(adata.var_names)
    print(f"   ✅ Gene names preprocessed (uppercase, Ensembl stripped)")
    
    # Load housekeeping genes
    housekeeping = load_housekeeping(args.housekeeping)
    if housekeeping:
        print(f"   ✅ Loaded {len(housekeeping)} housekeeping genes to exclude")
    
    # Build spot texts
    spot_texts = build_spot_texts(adata, topn=args.topn, housekeeping=housekeeping)
    
    # Load model
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    if args.use_demo:
        print(f"\n🤖 Loading lightweight demo checkpoint (no HF weights)")
        print(f"   [INFO] Using demo mode for testing/validation")
        try:
            model, preprocess, tokenizer = loki.utils.load_model(None, device)
            print(f"   ✅ Demo model loaded on {device}")
        except Exception as e:
            print(f"   ❌ Demo checkpoint load failed: {e}")
            print(f"   Check that Loki is installed via 'pip install .' from repo root")
            raise
    else:
        if args.hf_ckpt is None:
            raise ValueError("Either --hf_ckpt or --use_demo must be specified")
        print(f"\n🤖 Loading OmiCLIP model: {args.hf_ckpt}")
        print(f"   [INFO] Using HF checkpoint: {args.hf_ckpt}")
        model, preprocess, tokenizer = loki.utils.load_model(args.hf_ckpt, device)
        print(f"   ✅ Model loaded on {device}")
    
    # Encode spot texts
    text_embeddings = encode_spot_texts(
        model, tokenizer, spot_texts, device, batch_size=args.batch_size
    )
    
    # Extract expression matrix
    print("\n💾 Extracting expression matrix...")
    if hasattr(adata.X, 'toarray'):
        expr_matrix = adata.X.toarray()
    else:
        expr_matrix = adata.X
    
    print(f"   ✅ Expression matrix shape: {expr_matrix.shape}")
    
    # Save outputs
    print(f"\n💾 Saving bank files to {out_dir}...")
    
    # Save text embeddings
    np.save(out_dir / "bank_text_emb.npy", text_embeddings.astype(np.float32))
    print(f"   ✅ Saved: bank_text_emb.npy ({text_embeddings.shape})")
    
    # Save expression matrix
    np.save(out_dir / "bank_expr.npy", expr_matrix.astype(np.float32))
    print(f"   ✅ Saved: bank_expr.npy ({expr_matrix.shape})")
    
    # Save gene list
    with open(out_dir / "bank_genes.txt", 'w') as f:
        for gene in adata.var_names:
            f.write(f"{gene}\n")
    print(f"   ✅ Saved: bank_genes.txt ({len(adata.var_names)} genes)")
    
    # Save spot IDs
    with open(out_dir / "bank_spots.txt", 'w') as f:
        for spot in adata.obs_names:
            f.write(f"{spot}\n")
    print(f"   ✅ Saved: bank_spots.txt ({len(adata.obs_names)} spots)")
    
    # Save README
    readme_content = f"""# Text Bank for Zero-Shot Retrieval

## Generation Info

- **Source**: {args.bank_h5ad}
- **Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}
- **Spots**: {adata.shape[0]}
- **Genes**: {adata.shape[1]}
- **Top-N genes per spot**: {args.topn}
- **Min counts filter**: {args.min_counts}
- **Housekeeping excluded**: {len(housekeeping) if housekeeping else 0}
- **Embedding dimension**: {text_embeddings.shape[1]}

## Files

- `bank_text_emb.npy`: Text embeddings (spots × D), float32
- `bank_expr.npy`: Expression matrix (spots × genes), float32
- `bank_genes.txt`: Gene names (one per line, uppercase)
- `bank_spots.txt`: Spot IDs (one per line)

## Usage

```bash
python zero_shot_loki.py \\
  --tiles_dir /path/to/tiles \\
  --bank_text_emb npy:{out_dir}/bank_text_emb.npy \\
  --bank_expr npy:{out_dir}/bank_expr.npy \\
  --bank_genes {out_dir}/bank_genes.txt \\
  --out_csv pred_tile_gene.csv \\
  --hf_ckpt {args.hf_ckpt}
```

## Notes

- Gene names are uppercase with Ensembl versions stripped
- Text embeddings are L2-normalized
- Expression values are raw counts (not log-transformed)
"""
    
    with open(out_dir / "README_BANK.md", 'w') as f:
        f.write(readme_content)
    print(f"   ✅ Saved: README_BANK.md")
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"✅ Bank building completed in {elapsed:.1f}s")
    print("=" * 70)


if __name__ == '__main__':
    main()

