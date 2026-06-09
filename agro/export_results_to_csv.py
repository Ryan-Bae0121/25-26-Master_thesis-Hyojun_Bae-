#!/usr/bin/env python3

import os
import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Import Loki modules
from loki import utils
from loki import predex

print("=== Loki PredEx Results to CSV Export ===")

# Load actual OmiCLIP model
print("Loading OmiCLIP model...")
data_dir = '/home/students/hbae/Loki/data/basic_usage'
model_path = os.path.join(data_dir, 'checkpoint.pt')
device = 'cpu'

model, preprocess, tokenizer = utils.load_model(model_path, device)
model.eval()
print("OmiCLIP model loaded successfully!")

# Define ST sentences for different cell types
st_sentences = {
    'tumor': 'TP53 MYC EGFR KRAS PIK3CA PTEN AKT1 BRAF CDKN2A RB1 MDM2 CCND1 CCNE1 CDK4 CDK6 MYC BCL2 BCL6',
    'stroma': 'COL1A1 COL3A1 COL5A1 FN1 VIM ACTA2 TAGLN MYH11 PDGFRA PDGFRB FAP S100A4',
    'immune': 'CD3D CD3E CD3G CD4 CD8A CD19 CD20 MS4A1 CD68 CD163 CD11B ITGAM CD14',
    'endothelial': 'PECAM1 CD31 VWF FLT1 KDR VEGFA ANGPT1 ANGPT2 TEK TIE1',
    'epithelial': 'KRT8 KRT18 KRT19 EPCAM CDH1 CLDN3 CLDN4 CLDN7 TJP1',
    'normal': 'GAPDH ACTB TUBB B2M RPL13A RPS18 RPL32 RPS27A'
}

# Extract patches from HDF5 files
def extract_patches_from_hdf5(hdf5_path, max_patches=100):
    patches = []
    patch_names = []
    
    with h5py.File(hdf5_path, 'r') as f:
        patch_keys = list(f.keys())
        selected_keys = patch_keys[:max_patches]
        
        for key in selected_keys:
            patch_data = f[key][:]
            patch_image = Image.fromarray(patch_data)
            patches.append(patch_image)
            patch_names.append(key)
    
    return patches, patch_names

# Load patches from your custom data
patches_dir = '/home/students/hbae/Loki/patches_hdf5_with_annot'
sample_dirs = [d for d in os.listdir(patches_dir) if os.path.isdir(os.path.join(patches_dir, d))]

# Load patches from first sample
sample_name = sample_dirs[0]
sample_dir = os.path.join(patches_dir, sample_name)
hdf5_files = [f for f in os.listdir(sample_dir) if f.endswith('.hdf5')]

if hdf5_files:
    hdf5_file = hdf5_files[0]
    hdf5_path = os.path.join(sample_dir, hdf5_file)
    
    print(f"Extracting patches from: {sample_name}")
    patches, patch_names = extract_patches_from_hdf5(hdf5_path, max_patches=50)
    
    print(f"Extracted {len(patches)} patches")

# Custom function to encode PIL Images directly
def encode_pil_images(model, preprocess, pil_images, device):
    image_embeddings = []
    
    for pil_image in pil_images:
        image_input = torch.stack([preprocess(pil_image)]).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(image_input)
        
        image_embeddings.append(image_features)
    
    image_embeddings = torch.cat(image_embeddings, dim=0)
    image_embeddings = torch.nn.functional.normalize(image_embeddings, p=2, dim=-1)
    
    return image_embeddings

# Encode patches
print("Encoding patches...")
image_embeddings = encode_pil_images(model, preprocess, patches, device)

# Encode ST sentences
print("Encoding ST sentences...")
text_sentences = list(st_sentences.values())
text_embeddings = utils.encode_texts(
    model=model,
    tokenizer=tokenizer,
    texts=text_sentences,
    device=device
)

# Calculate similarity
print("Calculating similarity...")
if torch.is_tensor(image_embeddings):
    image_embeddings = image_embeddings.cpu().numpy()
if torch.is_tensor(text_embeddings):
    text_embeddings = text_embeddings.cpu().numpy()

similarity_matrix = np.dot(image_embeddings, text_embeddings.T)

# Create training data
gene_list = [
    'TP53', 'MYC', 'EGFR', 'KRAS', 'PIK3CA', 'PTEN', 'AKT1', 'BRAF', 'CDKN2A', 'RB1',
    'MDM2', 'CCND1', 'CCNE1', 'CDK4', 'CDK6', 'BCL2', 'BCL6', 'COL1A1', 'COL3A1', 'COL5A1',
    'FN1', 'VIM', 'ACTA2', 'TAGLN', 'MYH11', 'PDGFRA', 'PDGFRB', 'FAP', 'S100A4', 'CD3D',
    'CD3E', 'CD3G', 'CD4', 'CD8A', 'CD19', 'CD20', 'MS4A1', 'CD68', 'CD163', 'CD11B',
    'ITGAM', 'CD14', 'PECAM1', 'VWF', 'FLT1', 'KDR', 'VEGFA', 'ANGPT1', 'ANGPT2', 'TEK',
    'TIE1', 'KRT8', 'KRT18', 'KRT19', 'EPCAM', 'CDH1', 'CLDN3', 'CLDN4', 'CLDN7', 'TJP1',
    'GAPDH', 'ACTB', 'TUBB', 'B2M', 'RPL13A', 'RPS18', 'RPL32', 'RPS27A'
]

cell_types = list(st_sentences.keys())
np.random.seed(42)
train_data = np.zeros((len(cell_types), len(gene_list)))

# Assign higher expression to relevant genes for each cell type
for i, cell_type in enumerate(cell_types):
    if cell_type == 'tumor':
        tumor_genes = ['TP53', 'MYC', 'EGFR', 'KRAS', 'PIK3CA', 'PTEN', 'AKT1', 'BRAF']
        for gene in tumor_genes:
            if gene in gene_list:
                idx = gene_list.index(gene)
                train_data[i, idx] = np.random.uniform(0.7, 1.0)
    
    elif cell_type == 'stroma':
        stroma_genes = ['COL1A1', 'COL3A1', 'COL5A1', 'FN1', 'VIM', 'ACTA2', 'TAGLN']
        for gene in stroma_genes:
            if gene in gene_list:
                idx = gene_list.index(gene)
                train_data[i, idx] = np.random.uniform(0.7, 1.0)
    
    elif cell_type == 'immune':
        immune_genes = ['CD3D', 'CD3E', 'CD3G', 'CD4', 'CD8A', 'CD19', 'CD20', 'CD68']
        for gene in immune_genes:
            if gene in gene_list:
                idx = gene_list.index(gene)
                train_data[i, idx] = np.random.uniform(0.7, 1.0)
    
    elif cell_type == 'endothelial':
        endo_genes = ['PECAM1', 'VWF', 'FLT1', 'KDR', 'VEGFA', 'ANGPT1', 'ANGPT2']
        for gene in endo_genes:
            if gene in gene_list:
                idx = gene_list.index(gene)
                train_data[i, idx] = np.random.uniform(0.7, 1.0)
    
    elif cell_type == 'epithelial':
        epi_genes = ['KRT8', 'KRT18', 'KRT19', 'EPCAM', 'CDH1', 'CLDN3', 'CLDN4']
        for gene in epi_genes:
            if gene in gene_list:
                idx = gene_list.index(gene)
                train_data[i, idx] = np.random.uniform(0.7, 1.0)
    
    elif cell_type == 'normal':
        housekeeping_genes = ['GAPDH', 'ACTB', 'TUBB', 'B2M', 'RPL13A', 'RPS18']
        for gene in housekeeping_genes:
            if gene in gene_list:
                idx = gene_list.index(gene)
                train_data[i, idx] = np.random.uniform(0.7, 1.0)
    
    train_data[i, :] += np.random.uniform(0.1, 0.3, len(gene_list))

# Predict gene expression
print("Predicting gene expression...")
n_patches, n_cell_types = similarity_matrix.shape
expanded_similarity = np.zeros((n_patches, len(gene_list)))

genes_per_cell_type = len(gene_list) // n_cell_types
for i, cell_type in enumerate(cell_types):
    start_idx = i * genes_per_cell_type
    end_idx = start_idx + genes_per_cell_type
    if i == n_cell_types - 1:
        end_idx = len(gene_list)
    
    expanded_similarity[:, start_idx:end_idx] = similarity_matrix[:, i:i+1]

predicted_gene_expr = predex.predict_st_gene_expr(
    image_text_similarity=expanded_similarity,
    train_data=train_data.T
)

# Create DataFrames for CSV export
print("Creating CSV files...")

# 1. Similarity Matrix CSV
similarity_df = pd.DataFrame(
    similarity_matrix,
    index=[f"patch_{i+1}_{patch_names[i]}" for i in range(len(patches))],
    columns=cell_types
)
similarity_df.to_csv('/home/students/hbae/Loki/docs/notebooks/similarity_matrix.csv')
print("✅ Similarity matrix saved to: similarity_matrix.csv")

# 2. Predicted Gene Expression CSV
predicted_df = pd.DataFrame(
    predicted_gene_expr,
    index=[f"patch_{i+1}_{patch_names[i]}" for i in range(len(patches))],
    columns=cell_types
)
predicted_df.to_csv('/home/students/hbae/Loki/docs/notebooks/predicted_gene_expression.csv')
print("✅ Predicted gene expression saved to: predicted_gene_expression.csv")

# 3. Image Embeddings CSV
image_embeddings_df = pd.DataFrame(
    image_embeddings,
    index=[f"patch_{i+1}_{patch_names[i]}" for i in range(len(patches))],
    columns=[f"embedding_dim_{i}" for i in range(image_embeddings.shape[1])]
)
image_embeddings_df.to_csv('/home/students/hbae/Loki/docs/notebooks/image_embeddings.csv')
print("✅ Image embeddings saved to: image_embeddings.csv")

# 4. Text Embeddings CSV
text_embeddings_df = pd.DataFrame(
    text_embeddings,
    index=cell_types,
    columns=[f"embedding_dim_{i}" for i in range(text_embeddings.shape[1])]
)
text_embeddings_df.to_csv('/home/students/hbae/Loki/docs/notebooks/text_embeddings.csv')
print("✅ Text embeddings saved to: text_embeddings.csv")

# 5. Summary Statistics CSV
summary_stats = pd.DataFrame({
    'Metric': ['Sample', 'Total_Patches', 'Cell_Types', 'Embedding_Dim', 'Similarity_Min', 'Similarity_Max', 'Prediction_Min', 'Prediction_Max'],
    'Value': [sample_name, len(patches), len(cell_types), image_embeddings.shape[1], 
              similarity_matrix.min(), similarity_matrix.max(), 
              predicted_gene_expr.min(), predicted_gene_expr.max()]
})
summary_stats.to_csv('/home/students/hbae/Loki/docs/notebooks/summary_statistics.csv', index=False)
print("✅ Summary statistics saved to: summary_statistics.csv")

# 6. Top Predictions CSV
top_predictions = []
for cell_type in cell_types:
    max_idx = predicted_df[cell_type].idxmax()
    max_val = predicted_df[cell_type].max()
    top_predictions.append({
        'Cell_Type': cell_type,
        'Top_Patch': max_idx,
        'Max_Expression': max_val
    })

top_predictions_df = pd.DataFrame(top_predictions)
top_predictions_df.to_csv('/home/students/hbae/Loki/docs/notebooks/top_predictions.csv', index=False)
print("✅ Top predictions saved to: top_predictions.csv")

print("\n🎉 All CSV files created successfully!")
print(f"📁 Files saved in: /home/students/hbae/Loki/docs/notebooks/")
print("\n📊 Generated CSV files:")
print("   1. similarity_matrix.csv - Image-text similarity scores")
print("   2. predicted_gene_expression.csv - Predicted gene expression")
print("   3. image_embeddings.csv - Image embeddings (768 dimensions)")
print("   4. text_embeddings.csv - Text embeddings (768 dimensions)")
print("   5. summary_statistics.csv - Overall statistics")
print("   6. top_predictions.csv - Top predictions per cell type")
