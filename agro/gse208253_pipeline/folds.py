"""
Module for creating cross-validation folds
Uses GroupKFold to prevent data leakage at patient/sample level
"""

import numpy as np
import pandas as pd
import json
from sklearn.model_selection import GroupKFold
from typing import Dict, List, Tuple
from pathlib import Path


def create_group_folds(
    sample_ids: List[str],
    spot_counts: Dict[str, int],
    n_splits: int = 10,
    random_state: int = 42
) -> List[Tuple[List[str], List[str]]]:
    """
    Create group-based folds where entire samples go into train or val
    
    Args:
        sample_ids: List of sample identifiers
        spot_counts: Dictionary mapping sample_id to number of spots
        n_splits: Number of folds
        random_state: Random seed
        
    Returns:
        fold_splits: List of (train_samples, val_samples) tuples
    """
    # Create a dummy array for GroupKFold (one entry per sample)
    X = np.arange(len(sample_ids)).reshape(-1, 1)
    groups = np.arange(len(sample_ids))  # Each sample is its own group
    
    # Use GroupKFold
    gkf = GroupKFold(n_splits=n_splits)
    
    fold_splits = []
    
    for train_idx, val_idx in gkf.split(X, groups=groups):
        train_samples = [sample_ids[i] for i in train_idx]
        val_samples = [sample_ids[i] for i in val_idx]
        fold_splits.append((train_samples, val_samples))
    
    return fold_splits


def verify_no_leakage(fold_splits: List[Tuple[List[str], List[str]]]) -> bool:
    """
    Verify that no sample appears in both train and val within any fold
    and that no sample appears in multiple folds' val sets
    
    Args:
        fold_splits: List of (train_samples, val_samples) tuples
        
    Returns:
        is_valid: Whether there is no leakage
    """
    # Check within-fold leakage
    for i, (train, val) in enumerate(fold_splits):
        train_set = set(train)
        val_set = set(val)
        
        overlap = train_set & val_set
        if overlap:
            print(f"ERROR: Fold {i} has samples in both train and val: {overlap}")
            return False
    
    # Check cross-fold validation set overlap
    all_val_samples = []
    for i, (train, val) in enumerate(fold_splits):
        for sample in val:
            if sample in all_val_samples:
                print(f"ERROR: Sample {sample} appears in multiple validation sets")
                return False
            all_val_samples.append(sample)
    
    print(f"✓ No data leakage detected across {len(fold_splits)} folds")
    return True


def create_fold_mapping(
    adata_dict: Dict[str, any],
    n_splits: int = 10,
    random_state: int = 42
) -> Dict[int, Dict[str, List[str]]]:
    """
    Create fold mapping for samples
    
    Args:
        adata_dict: Dictionary of sample_id -> AnnData (or any object with n_obs)
        n_splits: Number of folds
        random_state: Random seed
        
    Returns:
        fold_mapping: Dictionary of fold_id -> {'train': [...], 'val': [...]}
    """
    sample_ids = list(adata_dict.keys())
    spot_counts = {sid: adata_dict[sid].n_obs for sid in sample_ids}
    
    # Create folds
    fold_splits = create_group_folds(sample_ids, spot_counts, n_splits, random_state)
    
    # Verify no leakage
    if not verify_no_leakage(fold_splits):
        raise ValueError("Data leakage detected in fold creation!")
    
    # Convert to mapping format
    fold_mapping = {}
    for i, (train_samples, val_samples) in enumerate(fold_splits):
        fold_mapping[i] = {
            'train': train_samples,
            'val': val_samples
        }
    
    # Print fold statistics
    print("\nFold Statistics:")
    for fold_id, splits in fold_mapping.items():
        n_train = sum(spot_counts[s] for s in splits['train'])
        n_val = sum(spot_counts[s] for s in splits['val'])
        print(f"  Fold {fold_id}: {len(splits['train'])} train samples ({n_train} spots), "
              f"{len(splits['val'])} val samples ({n_val} spots)")
    
    return fold_mapping


def save_folds(fold_mapping: Dict, output_path: Path) -> None:
    """
    Save fold mapping to JSON
    
    Args:
        fold_mapping: Fold mapping dictionary
        output_path: Output JSON file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert keys to strings for JSON
    fold_mapping_str = {str(k): v for k, v in fold_mapping.items()}
    
    with open(output_path, 'w') as f:
        json.dump(fold_mapping_str, f, indent=2)
    
    print(f"Saved fold mapping to {output_path}")


def load_folds(input_path: Path) -> Dict[int, Dict[str, List[str]]]:
    """
    Load fold mapping from JSON
    
    Args:
        input_path: Input JSON file path
        
    Returns:
        fold_mapping: Fold mapping dictionary with integer keys
    """
    with open(input_path, 'r') as f:
        fold_mapping_str = json.load(f)
    
    # Convert keys back to integers
    fold_mapping = {int(k): v for k, v in fold_mapping_str.items()}
    
    return fold_mapping


def get_fold_sample_ids(
    fold_mapping: Dict[int, Dict[str, List[str]]],
    fold_id: int,
    split: str = 'train'
) -> List[str]:
    """
    Get sample IDs for a specific fold and split
    
    Args:
        fold_mapping: Fold mapping dictionary
        fold_id: Fold index
        split: 'train' or 'val'
        
    Returns:
        sample_ids: List of sample IDs
    """
    return fold_mapping[fold_id][split]


def assign_spots_to_folds(
    combined_obs: pd.DataFrame,
    fold_mapping: Dict[int, Dict[str, List[str]]]
) -> pd.DataFrame:
    """
    Assign fold labels to each spot in combined_obs
    
    Args:
        combined_obs: Combined observation DataFrame with 'sample_id' column
        fold_mapping: Fold mapping dictionary
        
    Returns:
        combined_obs: DataFrame with 'fold_id' and 'split' columns added
    """
    combined_obs = combined_obs.copy()
    
    # Initialize columns
    combined_obs['fold_id'] = -1
    combined_obs['split'] = ''
    
    # Assign folds
    for fold_id, splits in fold_mapping.items():
        for split_name, sample_ids in splits.items():
            mask = combined_obs['sample_id'].isin(sample_ids)
            combined_obs.loc[mask, 'fold_id'] = fold_id
            combined_obs.loc[mask, 'split'] = split_name
    
    return combined_obs



