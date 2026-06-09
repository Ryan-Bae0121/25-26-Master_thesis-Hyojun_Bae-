#!/usr/bin/env python3
"""
Zero-shot gene expression prediction using Loki/OmiCLIP foundation model.
Uses retrieval-based inference with a Visium text bank.
"""

import argparse
import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
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


def load_bank_data(bank_text_emb, bank_expr, bank_genes):
    """Load bank embeddings, expression, and genes"""
    print("📂 Loading bank data...")
    
    # Load text embeddings
    if bank_text_emb.startswith('npy:'):
        text_emb = np.load(bank_text_emb[4:])
    elif bank_text_emb.startswith('pt:'):
        text_emb = torch.load(bank_text_emb[3:]).numpy()
    else:
        text_emb = np.load(bank_text_emb)
    
    print(f"   ✅ Text embeddings: {text_emb.shape}")
    
    # Load expression
    if bank_expr.startswith('npy:'):
        expr = np.load(bank_expr[4:])
    elif bank_expr.startswith('pt:'):
        expr = torch.load(bank_expr[3:]).numpy()
    else:
        expr = np.load(bank_expr)
    
    print(f"   ✅ Expression matrix: {expr.shape}")
    
    # Load genes
    with open(bank_genes, 'r') as f:
        genes = [line.strip() for line in f if line.strip()]
    
    print(f"   ✅ Genes: {len(genes)}")
    
    # Validate dimensions
    if text_emb.shape[0] != expr.shape[0]:
        raise ValueError(f"Mismatch: text_emb has {text_emb.shape[0]} spots but expr has {expr.shape[0]}")
    
    if expr.shape[1] != len(genes):
        raise ValueError(f"Mismatch: expr has {expr.shape[1]} genes but gene list has {len(genes)}")
    
    return text_emb, expr, genes


def filter_genes(bank_expr, bank_genes, genes_list):
    """Filter to subset of genes"""
    if genes_list is None:
        return bank_expr, bank_genes
    
    print(f"🔍 Filtering to gene subset: {genes_list}")
    
    # Load target genes
    with open(genes_list, 'r') as f:
        target_genes = {line.strip().upper() for line in f if line.strip()}
    
    print(f"   Target genes: {len(target_genes)}")
    
    # Find intersection
    bank_genes_upper = [g.upper() for g in bank_genes]
    keep_indices = [i for i, g in enumerate(bank_genes_upper) if g in target_genes]
    kept_genes = [bank_genes[i] for i in keep_indices]
    
    print(f"   ✅ Kept {len(kept_genes)} genes (intersection)")
    
    if len(kept_genes) < 100:
        print(f"   ⚠️  WARNING: Only {len(kept_genes)} genes kept (< 100)")
    
    # Filter expression
    filtered_expr = bank_expr[:, keep_indices]
    
    return filtered_expr, kept_genes


def normalize_bank(bank_expr, mode):
    """Apply normalization to bank expression"""
    if mode == 'none':
        return bank_expr
    elif mode == 'bank_log1p':
        print("   Applying log1p to bank expression...")
        return np.log1p(bank_expr)
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")


def find_tiles(tiles_dir):
    """Find all tile images recursively or HDF5 file"""
    tiles_dir = Path(tiles_dir)
    
    # Check if it's an HDF5 file
    if tiles_dir.is_file() and tiles_dir.suffix in ['.hdf5', '.h5']:
        return tiles_dir
    
    # Check if directory contains HDF5 file
    if tiles_dir.is_dir():
        hdf5_files = list(tiles_dir.glob('*.hdf5')) + list(tiles_dir.glob('*.h5'))
        if hdf5_files:
            return hdf5_files[0]
    
    # Otherwise, find image files
    patterns = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
    tiles = []
    
    for pattern in patterns:
        tiles.extend(tiles_dir.rglob(pattern))
    
    tiles = sorted(tiles)
    return tiles


def encode_tiles(model, preprocess, tile_source, device, batch_size=128):
    """Encode tiles to image embeddings (from files or HDF5)"""
    import h5py
    
    # Check if HDF5
    if isinstance(tile_source, Path) and tile_source.suffix in ['.hdf5', '.h5']:
        print(f"🖼️  Encoding tiles from HDF5: {tile_source}")
        
        with h5py.File(tile_source, 'r') as f:
            tile_keys = list(f.keys())
            n_tiles = len(tile_keys)
            print(f"   Found {n_tiles} tiles in HDF5")
            
            all_embeddings = []
            
            for i in tqdm(range(0, n_tiles, batch_size), desc="Encoding tiles"):
                batch_keys = tile_keys[i:i+batch_size]
                
                # Load images from HDF5
                images = []
                for key in batch_keys:
                    try:
                        img_array = f[key][:]  # (256, 256, 3) uint8
                        img = Image.fromarray(img_array)
                        images.append(preprocess(img))
                    except Exception as e:
                        print(f"   ⚠️  Failed to load tile {key}: {e}")
                        images.append(torch.zeros(3, 224, 224))
                
                # Stack and move to device
                batch_tensor = torch.stack(images).to(device)
                
                # Encode
                with torch.no_grad():
                    embeddings = model.encode_image(batch_tensor)
                    embeddings = F.normalize(embeddings, p=2, dim=-1)
                
                all_embeddings.append(embeddings.cpu().numpy())
            
            embeddings = np.vstack(all_embeddings)
            print(f"   ✅ Image embeddings shape: {embeddings.shape}")
            
            return embeddings, tile_keys
    
    # Otherwise, load from file paths
    else:
        tile_paths = tile_source
        print(f"🖼️  Encoding {len(tile_paths)} tiles...")
        
        all_embeddings = []
        
        for i in tqdm(range(0, len(tile_paths), batch_size), desc="Encoding tiles"):
            batch_paths = tile_paths[i:i+batch_size]
            
            # Load and preprocess images
            images = []
            for path in batch_paths:
                try:
                    img = Image.open(path).convert('RGB')
                    images.append(preprocess(img))
                except Exception as e:
                    print(f"   ⚠️  Failed to load {path}: {e}")
                    # Use dummy image
                    images.append(torch.zeros(3, 224, 224))
            
            # Stack and move to device
            batch_tensor = torch.stack(images).to(device)
            
            # Encode
            with torch.no_grad():
                embeddings = model.encode_image(batch_tensor)
                embeddings = F.normalize(embeddings, p=2, dim=-1)
            
            all_embeddings.append(embeddings.cpu().numpy())
        
        embeddings = np.vstack(all_embeddings)
        print(f"   ✅ Image embeddings shape: {embeddings.shape}")
        
        return embeddings, tile_paths


def compute_predictions(tile_emb, bank_text_emb, bank_expr, temp=1.0, topk=64):
    """Compute predictions via retrieval aggregation"""
    print(f"🔮 Computing predictions (temp={temp}, topk={topk})...")
    
    # Compute similarities (tiles × spots)
    similarities = tile_emb @ bank_text_emb.T  # (N_tiles, N_spots)
    print(f"   Similarity matrix: {similarities.shape}")
    
    # Apply temperature
    similarities = similarities / temp
    
    # Top-k masking
    if topk > 0 and topk < similarities.shape[1]:
        print(f"   Applying top-{topk} masking...")
        # Get top-k indices per tile
        topk_indices = np.argsort(similarities, axis=1)[:, -topk:]
        
        # Create mask
        mask = np.zeros_like(similarities)
        for i in range(similarities.shape[0]):
            mask[i, topk_indices[i]] = 1
        
        # Apply mask
        similarities = similarities * mask
    
    # Softmax weights
    exp_sim = np.exp(similarities - similarities.max(axis=1, keepdims=True))
    weights = exp_sim / exp_sim.sum(axis=1, keepdims=True)
    
    # Weighted aggregation
    predictions = weights @ bank_expr  # (N_tiles, N_genes)
    
    print(f"   ✅ Predictions shape: {predictions.shape}")
    
    return predictions


def save_results(predictions, tile_identifiers, genes, out_csv, out_dir, info):
    """Save prediction results"""
    print(f"💾 Saving results to {out_csv}...")
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Create DataFrame
    # tile_identifiers can be paths or HDF5 keys
    if isinstance(tile_identifiers[0], (str, Path)):
        if isinstance(tile_identifiers[0], Path):
            tile_names = [p.name for p in tile_identifiers]
        else:
            tile_names = tile_identifiers  # HDF5 keys
    else:
        tile_names = [str(t) for t in tile_identifiers]
    
    df = pd.DataFrame(predictions, index=tile_names, columns=genes)
    
    # Save CSV
    df.to_csv(out_csv, float_format='%.6f')
    print(f"   ✅ Saved: {out_csv}")
    
    # Save tile index
    tile_index_path = out_dir / "tile_index.tsv"
    with open(tile_index_path, 'w') as f:
        f.write("tile_name\ttile_identifier\n")
        for i, identifier in enumerate(tile_identifiers):
            f.write(f"{tile_names[i]}\t{identifier}\n")
    print(f"   ✅ Saved: {tile_index_path}")
    
    # Save gene index
    gene_index_path = out_dir / "gene_index.tsv"
    with open(gene_index_path, 'w') as f:
        f.write("gene\n")
        for gene in genes:
            f.write(f"{gene}\n")
    print(f"   ✅ Saved: {gene_index_path}")
    
    # Save info
    info_path = out_dir / "pred_info.json"
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"   ✅ Saved: {info_path}")


def main():
    parser = argparse.ArgumentParser(description='Zero-shot gene prediction with Loki')
    
    # Input/output
    parser.add_argument('--tiles_dir', type=str, required=True,
                       help='Directory containing tiles or HDF5 file path')
    parser.add_argument('--bank_text_emb', type=str, required=True,
                       help='Bank text embeddings (npy:/path or pt:/path)')
    parser.add_argument('--bank_expr', type=str, required=True,
                       help='Bank expression matrix (npy:/path or pt:/path)')
    parser.add_argument('--bank_genes', type=str, required=True,
                       help='Bank gene list (txt file)')
    parser.add_argument('--out_csv', type=str, required=True,
                       help='Output CSV path')
    parser.add_argument('--hf_ckpt', type=str, default=None,
                       help='Path to OmiCLIP checkpoint.pt (not needed if --use_demo)')
    
    # Options
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for encoding')
    parser.add_argument('--genes_list', type=str, default=None,
                       help='Optional: restrict to gene subset')
    parser.add_argument('--normalize', type=str, default='bank_log1p',
                       choices=['none', 'bank_log1p'],
                       help='Normalization mode')
    parser.add_argument('--temp', type=float, default=1.0,
                       help='Softmax temperature')
    parser.add_argument('--topk', type=int, default=64,
                       help='Top-k spots per tile (0=all)')
    parser.add_argument('--use_demo', action='store_true',
                       help='Use lightweight demo checkpoint (no HF weights needed)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Zero-Shot Gene Expression Prediction (Loki Foundation)")
    print("=" * 70)
    
    start_time = time.time()
    
    # Load bank data
    bank_text_emb, bank_expr, bank_genes = load_bank_data(
        args.bank_text_emb, args.bank_expr, args.bank_genes
    )
    
    # Filter genes if needed
    if args.genes_list:
        bank_expr, bank_genes = filter_genes(bank_expr, bank_genes, args.genes_list)
    
    # Normalize bank
    print(f"\n🔧 Normalization mode: {args.normalize}")
    bank_expr = normalize_bank(bank_expr, args.normalize)
    
    # Find tiles
    print(f"\n📂 Finding tiles in: {args.tiles_dir}")
    tile_source = find_tiles(args.tiles_dir)
    
    # Check if HDF5 or file list
    if isinstance(tile_source, Path) and tile_source.suffix in ['.hdf5', '.h5']:
        import h5py
        with h5py.File(tile_source, 'r') as f:
            n_tiles = len(f.keys())
        print(f"   ✅ Found HDF5 with {n_tiles} tiles")
    else:
        n_tiles = len(tile_source)
        print(f"   ✅ Found {n_tiles} tile files")
    
    if n_tiles == 0:
        raise ValueError(f"No tiles found in {args.tiles_dir}")
    
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
    
    # Encode tiles
    tile_emb, tile_identifiers = encode_tiles(model, preprocess, tile_source, device, args.batch_size)
    
    # Compute predictions
    predictions = compute_predictions(
        tile_emb, bank_text_emb, bank_expr, 
        temp=args.temp, topk=args.topk
    )
    
    # Save results
    out_dir = Path(args.out_csv).parent
    info = {
        'embedding_dim': int(tile_emb.shape[1]),
        'n_tiles': int(tile_emb.shape[0]),
        'n_spots': int(bank_text_emb.shape[0]),
        'n_genes': int(len(bank_genes)),
        'topk': int(args.topk),
        'temperature': float(args.temp),
        'normalize': args.normalize,
        'device': str(device),
        'batch_size': int(args.batch_size),
        'tile_source': str(args.tiles_dir)
    }
    
    save_results(predictions, tile_identifiers, bank_genes, args.out_csv, out_dir, info)
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"✅ Zero-shot prediction completed in {elapsed:.1f}s")
    print(f"   Output: {args.out_csv}")
    print("=" * 70)


if __name__ == '__main__':
    main()

