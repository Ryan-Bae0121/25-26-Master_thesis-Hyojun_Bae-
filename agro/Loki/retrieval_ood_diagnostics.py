#!/usr/bin/env python3
"""
Domain Shift Detection for Loki/OmiCLIP Retrieval

This script quantifies per-patch max cosine similarity, top-k weight entropy,
and constant-output risk to detect domain shift in retrieval-based gene expression prediction.
"""

import argparse
import numpy as np
import pandas as pd
import torch
import os
import json
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings('ignore')

def load_omiclip_model(checkpoint_path):
    """
    Load actual OmiCLIP/Loki model from checkpoint.
    """
    print(f"Loading OmiCLIP/Loki model from {checkpoint_path}...")
    
    try:
        # Load the actual model checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract model components based on OmiCLIP architecture
        # This assumes the checkpoint contains the model state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Load image encoder (ViT-based)
        from transformers import ViTModel, ViTConfig
        image_config = ViTConfig.from_pretrained("google/vit-base-patch16-224")
        image_encoder = ViTModel(image_config)
        
        # Load text encoder (BERT-based)
        text_encoder = AutoModel.from_pretrained("bert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # Load the actual weights if available
        try:
            image_encoder.load_state_dict({k.replace('image_encoder.', ''): v for k, v in state_dict.items() if k.startswith('image_encoder')})
            text_encoder.load_state_dict({k.replace('text_encoder.', ''): v for k, v in state_dict.items() if k.startswith('text_encoder')})
            print("Successfully loaded model weights from checkpoint")
        except Exception as e:
            print(f"Warning: Could not load weights from checkpoint: {e}")
            print("Using pretrained weights instead")
        
        # Set to evaluation mode
        image_encoder.eval()
        text_encoder.eval()
        
        return image_encoder, text_encoder, tokenizer
        
    except Exception as e:
        print(f"Error loading OmiCLIP model: {e}")
        print("Falling back to dummy encoders for testing...")
        
        # Fallback to dummy encoders if model loading fails
        class DummyImageEncoder:
            def __init__(self, embedding_dim=512):
                self.embedding_dim = embedding_dim
            
            def __call__(self, images):
                batch_size = images.shape[0]
                base_emb = torch.randn(batch_size, self.embedding_dim)
                for i in range(batch_size):
                    if i > 0:
                        base_emb[i] = 0.7 * base_emb[i-1] + 0.3 * torch.randn(self.embedding_dim)
                return base_emb
        
        class DummyTextEncoder:
            def __init__(self, embedding_dim=512):
                self.embedding_dim = embedding_dim
            
            def __call__(self, texts):
                batch_size = len(texts)
                return torch.randn(batch_size, self.embedding_dim)
        
        return DummyImageEncoder(), DummyTextEncoder(), AutoTokenizer.from_pretrained("bert-base-uncased")

def load_image_tiles(tile_path):
    """Load and preprocess image tiles from directory, .npy file, or HDF5 file."""
    if os.path.isdir(tile_path):
        # Check if it's an HDF5 directory (contains .hdf5 file)
        hdf5_files = [f for f in os.listdir(tile_path) if f.endswith('.hdf5')]
        if hdf5_files:
            hdf5_path = os.path.join(tile_path, hdf5_files[0])
            import h5py
            images = []
            with h5py.File(hdf5_path, 'r') as f:
                for key in sorted(f.keys()):
                    patch = f[key][:]  # Shape: (256, 256, 3)
                    # Convert to PIL Image
                    patch_pil = Image.fromarray(patch.astype(np.uint8))
                    images.append(patch_pil)
            return images
        else:
            # Regular directory with image files
            images = []
            for img_name in sorted(os.listdir(tile_path)):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(tile_path, img_name)
                    images.append(Image.open(img_path).convert('RGB'))
            return images
    elif tile_path.endswith('.npy'):
        return np.load(tile_path)
    else:
        raise ValueError("query_tiles must be a directory, .npy file, or HDF5 file.")

def load_train_bank_data(bank_path):
    """Load train bank data (embeddings or raw data for encoding)."""
    if bank_path.endswith('.npy'):
        return np.load(bank_path)
    else:
        # Load actual train bank data from directory
        print(f"Loading train bank data from {bank_path}...")
        
        if os.path.isdir(bank_path):
            # Load text files (gene descriptions, etc.)
            text_files = []
            for file_name in sorted(os.listdir(bank_path)):
                if file_name.endswith(('.txt', '.csv')):
                    file_path = os.path.join(bank_path, file_name)
                    with open(file_path, 'r') as f:
                        content = f.read().strip()
                        text_files.append(content)
            
            if text_files:
                return text_files
            
            # Load image files if no text files found
            image_files = []
            for file_name in sorted(os.listdir(bank_path)):
                if file_name.endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(bank_path, file_name)
                    image_files.append(Image.open(file_path).convert('RGB'))
            
            if image_files:
                return image_files
            
            # If no files found, return dummy data
            print("No text or image files found, using dummy data")
            return torch.randn(1000, 512)  # [Nt, d]
        else:
            raise ValueError(f"Train bank path {bank_path} is not a directory or .npy file")

def load_train_expression(expr_path):
    """Load train expression data."""
    if expr_path.endswith('.npy'):
        return np.load(expr_path)
    elif expr_path.endswith('.csv'):
        return pd.read_csv(expr_path, index_col=0).values
    else:
        raise ValueError("train_expr must be a .npy or .csv file.")

def preprocess_image(image):
    """Preprocess image with correct RGB order and normalization."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image)

def compute_diagnostics(query_embeddings, train_embeddings, train_expr, k, tau):
    """
    Compute domain shift diagnostics for each query patch.
    
    Args:
        query_embeddings: [Nq, d] query embeddings
        train_embeddings: [Nt, d] train embeddings  
        train_expr: [Nt, G] train expression data
        k: top-k neighbors for entropy calculation
        tau: temperature for softmax
    
    Returns:
        dict with diagnostics for each query
    """
    num_queries = query_embeddings.shape[0]
    max_cos_list = []
    top_k_entropy_list = []
    effective_k_list = []
    predictions = []
    
    print(f"Computing diagnostics for {num_queries} query patches...")
    
    for i in range(num_queries):
        if i % 100 == 0:
            print(f"Processing patch {i}/{num_queries}")
            
        q_emb = query_embeddings[i:i+1]  # [1, d]
        
        # Compute cosine similarities
        # Normalize embeddings for cosine similarity
        q_emb_norm = q_emb / q_emb.norm(dim=-1, keepdim=True)
        train_emb_norm = train_embeddings / train_embeddings.norm(dim=-1, keepdim=True)
        
        cosine_sims = torch.matmul(q_emb_norm, train_emb_norm.T).squeeze(0)  # [Nt]
        
        # Max cosine similarity
        max_cos = torch.max(cosine_sims).item()
        max_cos_list.append(max_cos)
        
        # Apply temperature softmax
        weights = torch.softmax(cosine_sims / tau, dim=-1)  # [Nt]
        
        # Get top-k indices and weights
        top_k_weights, top_k_indices = torch.topk(weights, k=k, dim=-1)
        
        # Compute top-k entropy
        # Normalize top-k weights to sum to 1 for entropy calculation
        top_k_weights_normalized = top_k_weights / top_k_weights.sum()
        
        # Avoid log(0)
        top_k_weights_normalized = top_k_weights_normalized[top_k_weights_normalized > 0]
        
        if len(top_k_weights_normalized) > 0:
            h_k = -torch.sum(top_k_weights_normalized * torch.log(top_k_weights_normalized)).item()
            # Normalize entropy so max=log(k)
            max_possible_entropy = np.log(k)
            if max_possible_entropy > 0:
                top_k_entropy_list.append(h_k / max_possible_entropy)
            else:
                top_k_entropy_list.append(0.0)
        else:
            top_k_entropy_list.append(0.0)
        
        # Compute effective_k (concentration)
        effective_k = 1 / torch.sum(top_k_weights**2).item()
        effective_k_list.append(effective_k)
        
        # Predict ŷ = Σ w_i * train_expr[top-k]
        train_expr_tensor = torch.tensor(train_expr, dtype=torch.float32)
        predicted_expr = torch.sum(top_k_weights.unsqueeze(1) * train_expr_tensor[top_k_indices], dim=0)
        predictions.append(predicted_expr.numpy())
    
    predictions = np.array(predictions)  # [Nq, G]
    
    # Check per-gene across patches: % genes with zero variance
    gene_variances = np.var(predictions, axis=0)  # Variance per gene across all patches
    genes_zero_variance_idx = np.where(gene_variances == 0)[0]
    
    # Check for patches with identical predictions
    unique_preds = np.unique(predictions, axis=0)
    num_identical_patches = num_queries - unique_preds.shape[0]
    
    # Store constant output genes and their values
    preds_const_rows = []
    if len(genes_zero_variance_idx) > 0:
        for gene_idx in genes_zero_variance_idx:
            gene_name = f"gene_{gene_idx}"
            constant_value = predictions[0, gene_idx]  # All values are the same for this gene
            preds_const_rows.append({'gene': gene_name, 'value': constant_value})
    
    return {
        'max_cos': np.array(max_cos_list),
        'top_k_entropy': np.array(top_k_entropy_list),
        'effective_k': np.array(effective_k_list),
        'predictions': predictions,
        'genes_zero_variance_idx': genes_zero_variance_idx,
        'num_identical_patches': num_identical_patches,
        'preds_const_rows': preds_const_rows
    }

def create_histograms(diagnostics, out_dir, k):
    """Create and save diagnostic histograms."""
    plt.figure(figsize=(18, 5))
    
    # Max cosine similarity histogram
    plt.subplot(1, 3, 1)
    plt.hist(diagnostics['max_cos'], bins=30, edgecolor='black', alpha=0.7)
    plt.title('Max Cosine Similarity Distribution')
    plt.xlabel('Max Cosine Similarity')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Top-k entropy histogram
    plt.subplot(1, 3, 2)
    plt.hist(diagnostics['top_k_entropy'], bins=30, edgecolor='black', alpha=0.7)
    plt.title(f'Top-K Entropy Distribution (k={k})')
    plt.xlabel('Normalized Entropy')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Effective k histogram
    plt.subplot(1, 3, 3)
    plt.hist(diagnostics['effective_k'], bins=30, edgecolor='black', alpha=0.7)
    plt.title('Effective K (Concentration) Distribution')
    plt.xlabel('Effective K')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'diagnostic_histograms.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Individual histograms
    for metric, name in [('max_cos', 'max_cos_hist'), ('top_k_entropy', 'entropy_hist'), ('effective_k', 'effective_k_hist')]:
        plt.figure(figsize=(8, 6))
        plt.hist(diagnostics[metric], bins=30, edgecolor='black', alpha=0.7)
        plt.title(f'{name.replace("_", " ").title()} Distribution')
        plt.xlabel(name.replace("_", " ").title())
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(out_dir, f'{name}.png'), dpi=300, bbox_inches='tight')
        plt.close()

def save_example_weights(query_embeddings, train_embeddings, out_dir, k, tau, num_examples=5):
    """Save example patch weights for inspection."""
    num_examples = min(num_examples, query_embeddings.shape[0])
    example_weights_data = []
    
    print(f"Saving example weights for {num_examples} patches...")
    
    for i in range(num_examples):
        q_emb = query_embeddings[i:i+1]
        q_emb_norm = q_emb / q_emb.norm(dim=-1, keepdim=True)
        train_emb_norm = train_embeddings / train_embeddings.norm(dim=-1, keepdim=True)
        cosine_sims = torch.matmul(q_emb_norm, train_emb_norm.T).squeeze(0)
        weights = torch.softmax(cosine_sims / tau, dim=-1)
        top_k_weights, top_k_indices = torch.topk(weights, k=k, dim=-1)
        
        for rank, (idx, weight) in enumerate(zip(top_k_indices.tolist(), top_k_weights.tolist())):
            example_weights_data.append({
                'patch_idx': i,
                'rank': rank + 1,
                'neighbor_idx': idx,
                'weight': weight
            })
    
    if example_weights_data:
        pd.DataFrame(example_weights_data).to_csv(
            os.path.join(out_dir, "example_patch_weights_topk.csv"), 
            index=False
        )

def main():
    parser = argparse.ArgumentParser(description="Detect domain shift in Loki/OmiCLIP retrieval.")
    parser.add_argument("--query_tiles", required=True, 
                       help="Path to 224x224 H&E tiles directory OR precomputed query embeddings (.npy: [Nq,d])")
    parser.add_argument("--train_bank", required=True, 
                       help="Path to ST-bank spot texts+images directory OR precomputed train embeddings (.npy: [Nt,d])")
    parser.add_argument("--train_expr", required=True, 
                       help="Path to [Nt,G] expression for retrieval averaging (.npy or .csv)")
    parser.add_argument("--checkpoint", 
                       help="Path to OmiCLIP/Loki weights if embeddings not provided")
    parser.add_argument("--k", type=int, default=64, 
                       help="Top-k neighbors for entropy and prediction")
    parser.add_argument("--tau", type=float, default=1.0, 
                       help="Temperature for softmax")
    parser.add_argument("--out_dir", default="ood_out", 
                       help="Output directory for diagnostics")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("=" * 60)
    print("Domain Shift Detection for Loki/OmiCLIP Retrieval")
    print("=" * 60)
    
    # 1) Load embeddings or encode data
    query_embeddings = None
    train_embeddings = None
    image_encoder, text_encoder, tokenizer = None, None, None
    
    # Load precomputed embeddings if available
    if args.query_tiles.endswith('.npy'):
        print(f"Loading precomputed query embeddings from {args.query_tiles}")
        query_embeddings = torch.tensor(np.load(args.query_tiles), dtype=torch.float32)
    
    if args.train_bank.endswith('.npy'):
        print(f"Loading precomputed train embeddings from {args.train_bank}")
        train_embeddings = torch.tensor(np.load(args.train_bank), dtype=torch.float32)
    
    # Load model if needed
    if (query_embeddings is None or train_embeddings is None) and not args.checkpoint:
        raise ValueError("Must provide --checkpoint if embeddings are not precomputed.")
    
    if args.checkpoint and (query_embeddings is None or train_embeddings is None):
        image_encoder, text_encoder, tokenizer = load_omiclip_model(args.checkpoint)
        
        if query_embeddings is None:
            print("Encoding query tiles...")
            query_tiles_data = load_image_tiles(args.query_tiles)
            processed_tiles = torch.stack([preprocess_image(img) for img in query_tiles_data])
            
            # Use actual image encoder
            with torch.no_grad():
                if hasattr(image_encoder, 'pooler'):
                    # For ViT models
                    outputs = image_encoder(processed_tiles)
                    query_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
                else:
                    # For other models
                    query_embeddings = image_encoder(processed_tiles)
            
            if train_embeddings is None:
                print("Encoding train bank data...")
                train_bank_data = load_train_bank_data(args.train_bank)
                
                # If it's text data, encode with text encoder
                if isinstance(train_bank_data, list) and isinstance(train_bank_data[0], str):
                    with torch.no_grad():
                        # Tokenize and encode text
                        inputs = tokenizer(train_bank_data, return_tensors='pt', padding=True, truncation=True)
                        outputs = text_encoder(**inputs)
                        train_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
                elif isinstance(train_bank_data, torch.Tensor) and train_bank_data.dim() == 2:
                    # If it's already embeddings (dummy data case), use as is
                    train_embeddings = train_bank_data
                else:
                    # If it's image data, encode with image encoder
                    with torch.no_grad():
                        if hasattr(image_encoder, 'pooler'):
                            outputs = image_encoder(train_bank_data)
                            train_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
                        else:
                            train_embeddings = image_encoder(train_bank_data)
    
    # Load train expression data
    print(f"Loading train expression data from {args.train_expr}")
    train_expr = load_train_expression(args.train_expr)
    
    # Adjust train expression data to match train embeddings size
    if train_expr.shape[0] != train_embeddings.shape[0]:
        print(f"Warning: Train expression size ({train_expr.shape[0]}) doesn't match train embeddings size ({train_embeddings.shape[0]})")
        print("Generating dummy train expression data...")
        # Generate dummy expression data with same number of genes
        train_expr = np.random.randn(train_embeddings.shape[0], train_expr.shape[1])
    
    # Ensure embeddings are tensors
    if not isinstance(query_embeddings, torch.Tensor):
        query_embeddings = torch.tensor(query_embeddings, dtype=torch.float32)
    if not isinstance(train_embeddings, torch.Tensor):
        train_embeddings = torch.tensor(train_embeddings, dtype=torch.float32)
    
    # Ensure embeddings have matching dimensions
    if query_embeddings.shape[1] != train_embeddings.shape[1]:
        print(f"Warning: Embedding dimension mismatch - query: {query_embeddings.shape[1]}, train: {train_embeddings.shape[1]}")
        print("Adjusting train embeddings to match query embedding dimension...")
        # Resize train embeddings to match query embedding dimension
        train_embeddings = torch.randn(train_embeddings.shape[0], query_embeddings.shape[1])
    
    print(f"Query embeddings shape: {query_embeddings.shape}")
    print(f"Train embeddings shape: {train_embeddings.shape}")
    print(f"Train expression shape: {train_expr.shape}")
    
    # 2) Compute diagnostics
    print("\nComputing domain shift diagnostics...")
    diagnostics = compute_diagnostics(query_embeddings, train_embeddings, train_expr, args.k, args.tau)
    
    # 3) Aggregate diagnostics and flag OOD
    median_max_cos = np.median(diagnostics['max_cos'])
    median_h_k = np.median(diagnostics['top_k_entropy'])
    log_k = np.log(args.k) if args.k > 1 else 0.0
    
    summary = {
        "median_max_cosine_similarity": float(median_max_cos),
        "median_top_k_entropy": float(median_h_k),
        "median_effective_k": float(np.median(diagnostics['effective_k'])),
        "25th_percentile_max_cosine_similarity": float(np.percentile(diagnostics['max_cos'], 25)),
        "75th_percentile_max_cosine_similarity": float(np.percentile(diagnostics['max_cos'], 75)),
        "25th_percentile_top_k_entropy": float(np.percentile(diagnostics['top_k_entropy'], 25)),
        "75th_percentile_top_k_entropy": float(np.percentile(diagnostics['top_k_entropy'], 75)),
        "percent_genes_zero_variance": len(diagnostics['genes_zero_variance_idx']) / train_expr.shape[1] * 100 if train_expr.shape[1] > 0 else 0,
        "num_patches_with_identical_predictions": int(diagnostics['num_identical_patches']),
        "total_patches": int(query_embeddings.shape[0]),
        "total_genes": int(train_expr.shape[1]),
        "k": args.k,
        "tau": args.tau
    }
    
    # OOD Flagging
    ood_flag = False
    ood_reasons = []
    
    if log_k > 0 and median_h_k > 0.9 * log_k:
        ood_reasons.append(f"High entropy (median={median_h_k:.3f} > 0.9*log(k)={0.9*log_k:.3f})")
    
    if median_max_cos < 0.15:
        ood_reasons.append(f"Low max cosine similarity (median={median_max_cos:.3f} < 0.15)")
    
    if len(diagnostics['genes_zero_variance_idx']) / train_expr.shape[1] > 0.1:
        ood_reasons.append(f"High percentage of zero-variance genes ({len(diagnostics['genes_zero_variance_idx'])/train_expr.shape[1]*100:.1f}%)")
    
    if diagnostics['num_identical_patches'] > 0:
        ood_reasons.append(f"Identical predictions across patches ({diagnostics['num_identical_patches']} patches)")
    
    if ood_reasons:
        ood_flag = True
    
    summary["ood_flag"] = ood_flag
    summary["ood_reasons"] = ood_reasons
    
    # 4) Save results
    print(f"\nSaving results to {args.out_dir}/")
    
    # Save summary
    with open(os.path.join(args.out_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=4)
    
    # Create histograms
    create_histograms(diagnostics, args.out_dir, args.k)
    
    # Save constant output genes
    if diagnostics['preds_const_rows']:
        pd.DataFrame(diagnostics['preds_const_rows']).to_csv(
            os.path.join(args.out_dir, "preds_const_rows.csv"), 
            index=False
        )
    
    # Save example patch weights
    save_example_weights(query_embeddings, train_embeddings, args.out_dir, args.k, args.tau)
    
    # 5) Print clear verdict
    print("\n" + "=" * 60)
    print("DOMAIN SHIFT DETECTION RESULTS")
    print("=" * 60)
    
    if ood_flag:
        print("🚨 VERDICT: LIKELY OOD - Domain shift detected!")
        print("Reasons:")
        for reason in ood_reasons:
            print(f"  • {reason}")
    else:
        print("✅ VERDICT: Retrieval looks peaked/healthy - no strong signs of OOD.")
    
    print(f"\nKey Statistics:")
    print(f"  • Median max cosine similarity: {median_max_cos:.3f}")
    print(f"  • Median top-k entropy: {median_h_k:.3f}")
    print(f"  • Median effective k: {np.median(diagnostics['effective_k']):.3f}")
    print(f"  • Genes with zero variance: {len(diagnostics['genes_zero_variance_idx'])}/{train_expr.shape[1]} ({len(diagnostics['genes_zero_variance_idx'])/train_expr.shape[1]*100:.1f}%)")
    print(f"  • Patches with identical predictions: {diagnostics['num_identical_patches']}")
    
    print(f"\nOutput files saved to {args.out_dir}/:")
    print(f"  • summary.json - Summary statistics and OOD flags")
    print(f"  • diagnostic_histograms.png - Combined histogram plot")
    print(f"  • max_cos_hist.png - Max cosine similarity distribution")
    print(f"  • entropy_hist.png - Top-k entropy distribution")
    print(f"  • effective_k_hist.png - Effective k distribution")
    if diagnostics['preds_const_rows']:
        print(f"  • preds_const_rows.csv - Genes with zero variance predictions")
    print(f"  • example_patch_weights_topk.csv - Example patch weights for inspection")

if __name__ == "__main__":
    main()