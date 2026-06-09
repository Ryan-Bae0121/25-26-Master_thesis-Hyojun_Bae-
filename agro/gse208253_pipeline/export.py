"""
Module for exporting final training dataframe
Combines all metadata into a single CSV for model training
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import os


def create_training_dataframe(
    adata_dict: Dict[str, any],
    sentences_dict: Dict[str, pd.Series],
    dataset_name: str = "GSE208253",
    patient_id_col: str = "sample_id"
) -> pd.DataFrame:
    """
    Create training dataframe with all required fields
    
    Args:
        adata_dict: Dictionary of sample_id -> AnnData
        sentences_dict: Dictionary of sample_id -> gene sentences Series
        dataset_name: Name of the dataset
        patient_id_col: Column name for patient/sample ID
        
    Returns:
        train_df: DataFrame with columns [label, sample_name, dataset, img_path, patient_id]
    """
    rows = []
    
    for sample_id, adata in adata_dict.items():
        sentences = sentences_dict[sample_id]
        
        for barcode in adata.obs_names:
            # Get sentence (label)
            if barcode in sentences.index:
                label = sentences.loc[barcode]
            else:
                label = adata.obs.loc[barcode, 'gene_sentence']
            
            # Get image path
            img_path = adata.obs.loc[barcode, 'patch_path']
            
            # Extract patient ID (use sample_id as patient_id)
            if patient_id_col in adata.obs.columns:
                patient_id = adata.obs.loc[barcode, patient_id_col]
            else:
                patient_id = sample_id
            
            # Create row
            row = {
                'label': label,
                'sample_name': barcode,
                'dataset': dataset_name,
                'img_path': img_path,
                'patient_id': patient_id
            }
            
            rows.append(row)
    
    train_df = pd.DataFrame(rows)
    
    return train_df


def verify_training_dataframe(
    train_df: pd.DataFrame,
    check_files: bool = True,
    n_samples: int = 10
) -> bool:
    """
    Verify training dataframe integrity
    
    Args:
        train_df: Training dataframe
        check_files: Whether to check if image files exist
        n_samples: Number of random samples to check
        
    Returns:
        is_valid: Whether dataframe is valid
    """
    required_columns = ['label', 'sample_name', 'dataset', 'img_path', 'patient_id']
    
    # Check columns
    missing_cols = set(required_columns) - set(train_df.columns)
    if missing_cols:
        print(f"ERROR: Missing columns: {missing_cols}")
        return False
    
    # Check for nulls
    null_counts = train_df[required_columns].isnull().sum()
    if null_counts.any():
        print(f"ERROR: Null values found:\n{null_counts[null_counts > 0]}")
        return False
    
    # Check label format (should have ~50 genes)
    sample_labels = train_df['label'].sample(min(n_samples, len(train_df)))
    for label in sample_labels:
        n_genes = len(label.split())
        if n_genes < 10:  # At least 10 genes
            print(f"ERROR: Label has too few genes ({n_genes}): {label[:100]}...")
            return False
    
    # Check if image files exist
    if check_files:
        sample_paths = train_df['img_path'].sample(min(n_samples, len(train_df)))
        for img_path in sample_paths:
            if not os.path.exists(img_path):
                print(f"ERROR: Image file not found: {img_path}")
                return False
    
    print(f"✓ Training dataframe validated:")
    print(f"  - {len(train_df)} rows")
    print(f"  - {train_df['patient_id'].nunique()} unique patients/samples")
    print(f"  - {train_df['dataset'].nunique()} dataset(s)")
    print(f"  - Average genes per label: {train_df['label'].apply(lambda x: len(x.split())).mean():.1f}")
    
    return True


def add_fold_info_to_dataframe(
    train_df: pd.DataFrame,
    fold_mapping: Dict[int, Dict[str, List[str]]]
) -> pd.DataFrame:
    """
    Add fold information to training dataframe
    
    Args:
        train_df: Training dataframe
        fold_mapping: Fold mapping dictionary
        
    Returns:
        train_df: DataFrame with fold_id and split columns
    """
    train_df = train_df.copy()
    
    # Initialize columns
    train_df['fold_id'] = -1
    train_df['split'] = ''
    
    # Assign folds based on patient_id
    for fold_id, splits in fold_mapping.items():
        for split_name, sample_ids in splits.items():
            mask = train_df['patient_id'].isin(sample_ids)
            train_df.loc[mask, 'fold_id'] = fold_id
            train_df.loc[mask, 'split'] = split_name
    
    return train_df


def save_training_dataframe(
    train_df: pd.DataFrame,
    output_path: Path,
    verify: bool = True
) -> None:
    """
    Save training dataframe to CSV
    
    Args:
        train_df: Training dataframe
        output_path: Output CSV path
        verify: Whether to verify before saving
    """
    if verify:
        if not verify_training_dataframe(train_df, check_files=False):
            raise ValueError("Training dataframe validation failed!")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_path, index=False)
    
    print(f"\nSaved training dataframe to {output_path}")
    print(f"  Total spots: {len(train_df)}")
    print(f"  Columns: {', '.join(train_df.columns)}")


def verify_consistency_with_expression(
    train_df: pd.DataFrame,
    combined_expression: np.ndarray,
    common_genes: List[str]
) -> bool:
    """
    Verify consistency between train_df and combined expression matrix
    
    Args:
        train_df: Training dataframe
        combined_expression: Combined expression matrix
        common_genes: List of common genes
        
    Returns:
        is_consistent: Whether data is consistent
    """
    # Check number of rows
    if len(train_df) != combined_expression.shape[0]:
        print(f"ERROR: train_df has {len(train_df)} rows but expression has {combined_expression.shape[0]}")
        return False
    
    # Check gene count in labels
    n_genes_in_matrix = len(common_genes)
    sample_label = train_df['label'].iloc[0]
    genes_in_label = sample_label.split()
    
    # All genes in labels should be in common_genes
    common_genes_set = set(common_genes)
    invalid_genes = [g for g in genes_in_label if g not in common_genes_set]
    if invalid_genes:
        print(f"Warning: Some genes in labels not in common_genes: {invalid_genes[:5]}...")
    
    print(f"✓ Consistency check passed:")
    print(f"  - {len(train_df)} spots in both train_df and expression matrix")
    print(f"  - {n_genes_in_matrix} genes in expression matrix")
    
    return True


def export_complete_dataset(
    adata_dict: Dict[str, any],
    sentences_dict: Dict[str, pd.Series],
    fold_mapping: Dict[int, Dict[str, List[str]]],
    combined_expression: np.ndarray,
    common_genes: List[str],
    output_dir: Path,
    dataset_name: str = "GSE208253"
) -> pd.DataFrame:
    """
    Complete export pipeline
    
    Args:
        adata_dict: Dictionary of sample_id -> AnnData
        sentences_dict: Dictionary of sample_id -> sentences
        fold_mapping: Fold mapping
        combined_expression: Combined expression matrix
        common_genes: List of common genes
        output_dir: Output directory
        dataset_name: Dataset name
        
    Returns:
        train_df: Exported training dataframe
    """
    # Create training dataframe
    print("\nCreating training dataframe...")
    train_df = create_training_dataframe(adata_dict, sentences_dict, dataset_name)
    
    # Add fold information
    print("Adding fold information...")
    train_df = add_fold_info_to_dataframe(train_df, fold_mapping)
    
    # Verify consistency
    print("\nVerifying consistency...")
    verify_consistency_with_expression(train_df, combined_expression, common_genes)
    
    # Save
    output_path = output_dir / 'tables' / 'train_df.csv'
    save_training_dataframe(train_df, output_path, verify=True)
    
    return train_df



