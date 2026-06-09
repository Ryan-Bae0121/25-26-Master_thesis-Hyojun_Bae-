"""
Module for generating gene sentences from expression profiles
Creates space-separated gene lists based on top-k expressed genes
"""

import numpy as np
import pandas as pd
import anndata
from typing import List, Dict, Set, Optional
from collections import Counter


# Gene families to cap for TF-IDF style balancing
DEFAULT_GENE_FAMILIES = {
    'IGH': ['IGHG', 'IGHA', 'IGHM', 'IGHD', 'IGHE', 'IGHV', 'IGHJ', 'IGHD'],
    'IGK': ['IGKV', 'IGKJ', 'IGKC'],
    'IGL': ['IGLV', 'IGLJ', 'IGLC'],
    'RPL': ['RPL'],
    'RPS': ['RPS'],
    'S100': ['S100'],
    'MT-': ['MT-'],
}


def get_top_k_genes(
    expression_vector: np.ndarray,
    gene_names: List[str],
    k: int = 50
) -> List[str]:
    """
    Get top-k most highly expressed genes for a spot
    
    Args:
        expression_vector: Expression values for one spot (n_genes,)
        gene_names: List of gene names
        k: Number of top genes to select
        
    Returns:
        top_genes: List of top-k gene names
    """
    # Get indices of top k genes
    top_indices = np.argsort(expression_vector)[::-1][:k]
    top_genes = [gene_names[i] for i in top_indices]
    
    return top_genes


def apply_family_capping(
    gene_list: List[str],
    family_prefixes: Dict[str, List[str]],
    max_per_family: int = 10
) -> List[str]:
    """
    Cap the number of genes from each family (TF-IDF style)
    
    Args:
        gene_list: List of genes
        family_prefixes: Dictionary of family_name -> list of prefixes
        max_per_family: Maximum genes per family
        
    Returns:
        capped_genes: List with family capping applied
    """
    # Count genes per family
    family_counts = {family: 0 for family in family_prefixes}
    capped_genes = []
    
    for gene in gene_list:
        # Check which family this gene belongs to
        assigned_family = None
        for family, prefixes in family_prefixes.items():
            if any(gene.startswith(prefix) for prefix in prefixes):
                assigned_family = family
                break
        
        # Apply capping if in a family
        if assigned_family is not None:
            if family_counts[assigned_family] < max_per_family:
                capped_genes.append(gene)
                family_counts[assigned_family] += 1
        else:
            # Not in any family, always include
            capped_genes.append(gene)
    
    return capped_genes


def create_gene_sentence(
    expression_vector: np.ndarray,
    gene_names: List[str],
    k: int = 50,
    apply_tfidf_cap: bool = False,
    family_prefixes: Optional[Dict[str, List[str]]] = None,
    max_per_family: int = 10
) -> str:
    """
    Create a gene sentence from expression profile
    
    Args:
        expression_vector: Expression values for one spot
        gene_names: List of gene names
        k: Number of top genes to select
        apply_tfidf_cap: Whether to apply family capping
        family_prefixes: Dictionary of gene family prefixes
        max_per_family: Maximum genes per family
        
    Returns:
        sentence: Space-separated gene names
    """
    # Get top-k genes
    top_genes = get_top_k_genes(expression_vector, gene_names, k)
    
    # Apply family capping if requested
    if apply_tfidf_cap and family_prefixes is not None:
        top_genes = apply_family_capping(top_genes, family_prefixes, max_per_family)
    
    # Create sentence
    sentence = ' '.join(top_genes)
    
    return sentence


def create_sentences_for_sample(
    adata: anndata.AnnData,
    k: int = 50,
    layer: Optional[str] = None,
    apply_tfidf_cap: bool = False,
    family_prefixes: Optional[Dict[str, List[str]]] = None,
    max_per_family: int = 10
) -> pd.Series:
    """
    Create gene sentences for all spots in a sample
    
    Args:
        adata: AnnData object
        k: Number of top genes
        layer: Which layer to use (None for .X)
        apply_tfidf_cap: Apply family capping
        family_prefixes: Gene family prefixes
        max_per_family: Max genes per family
        
    Returns:
        sentences: Series with barcode index and gene sentences
    """
    # Get expression matrix
    if layer is None:
        X = adata.X
    else:
        X = adata.layers[layer]
    
    # Convert to dense if sparse
    if hasattr(X, 'toarray'):
        X = X.toarray()
    
    gene_names = list(adata.var_names)
    
    # Create sentences for each spot
    sentences = []
    for i in range(X.shape[0]):
        sentence = create_gene_sentence(
            X[i],
            gene_names,
            k=k,
            apply_tfidf_cap=apply_tfidf_cap,
            family_prefixes=family_prefixes,
            max_per_family=max_per_family
        )
        sentences.append(sentence)
    
    return pd.Series(sentences, index=adata.obs_names, name='gene_sentence')


def batch_create_sentences(
    adata_dict: Dict[str, anndata.AnnData],
    k: int = 50,
    layer: Optional[str] = None,
    apply_tfidf_cap: bool = False,
    cap_families: Optional[List[str]] = None,
    max_per_family: int = 10
) -> Dict[str, pd.Series]:
    """
    Create gene sentences for multiple samples
    
    Args:
        adata_dict: Dictionary of sample_id -> AnnData
        k: Number of top genes
        layer: Which layer to use
        apply_tfidf_cap: Apply family capping
        cap_families: List of family names to cap (e.g., ['IGH', 'RPL'])
        max_per_family: Max genes per family
        
    Returns:
        sentences_dict: Dictionary of sample_id -> sentences Series
    """
    # Parse family prefixes
    family_prefixes = None
    if apply_tfidf_cap and cap_families is not None:
        family_prefixes = {
            family: DEFAULT_GENE_FAMILIES.get(family, [family])
            for family in cap_families
        }
    
    sentences_dict = {}
    
    for sample_id, adata in adata_dict.items():
        print(f"Creating sentences for {sample_id}...")
        sentences = create_sentences_for_sample(
            adata,
            k=k,
            layer=layer,
            apply_tfidf_cap=apply_tfidf_cap,
            family_prefixes=family_prefixes,
            max_per_family=max_per_family
        )
        sentences_dict[sample_id] = sentences
    
    return sentences_dict


def add_sentences_to_adata(
    adata: anndata.AnnData,
    sentences: pd.Series
) -> anndata.AnnData:
    """
    Add gene sentences to AnnData obs
    
    Args:
        adata: AnnData object
        sentences: Series with gene sentences
        
    Returns:
        adata: Updated AnnData
    """
    adata.obs['gene_sentence'] = sentences.loc[adata.obs_names].values
    return adata


def verify_sentence_format(sentences: pd.Series, expected_k: int = 50) -> bool:
    """
    Verify that sentences have approximately the expected number of genes
    
    Args:
        sentences: Series with gene sentences
        expected_k: Expected number of genes (approximately)
        
    Returns:
        is_valid: Whether format is valid
    """
    # Check a few random sentences
    sample_sentences = sentences.sample(min(10, len(sentences)))
    
    for sentence in sample_sentences:
        genes = sentence.split()
        if len(genes) < expected_k * 0.5:  # Allow some variation due to capping
            print(f"Warning: Sentence has only {len(genes)} genes (expected ~{expected_k})")
            return False
    
    print(f"✓ Sentence format validated: ~{expected_k} genes per spot")
    return True



