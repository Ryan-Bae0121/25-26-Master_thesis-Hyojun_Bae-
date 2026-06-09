#!/bin/bash
# Evaluate predictions using top high-variance genes

set -e  # Exit on error

cd /project_antwerp/hbae/script/

# ============================================================================
# Configuration
# ============================================================================
PRED_FILE=./results_predex/Y_slide.npy
BULK_FILE=/project_antwerp/hbae/ref_file_for_eval.csv
GENE_NAMES=./results_predex/gene_names.npy
OUTPUT_DIR=./top_variance_eval_results

# K values to evaluate
K_VALUES="300 500 1000 2000"

echo "============================================================================"
echo "Top Variance Genes Evaluation"
echo "============================================================================"
echo "Prediction file: $PRED_FILE"
echo "Bulk file: $BULK_FILE"
echo "Gene names file: $GENE_NAMES"
echo "Output directory: $OUTPUT_DIR"
echo "K values: $K_VALUES"
echo "============================================================================"
echo ""

python 0208_start/TOP_300genes_test/evaluate_top_variance_genes.py \
    --pred_file $PRED_FILE \
    --bulk_file $BULK_FILE \
    --gene_names $GENE_NAMES \
    --output_dir $OUTPUT_DIR \
    --K_values $K_VALUES \
    --include_all \
    --verbose

echo ""
echo "============================================================================"
echo "Evaluation completed!"
echo "============================================================================"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Generated files:"
echo "  - top_variance_genes_evaluation.csv (comparison table)"
echo "  - comparison_across_K.png (comparison plots)"
echo "  - detailed_results.json (detailed metrics)"
echo "  - top_*_variance_gene_names.txt (selected gene lists)"
echo "  - gene_wise_pcc_distribution_K*.png (PCC distributions)"
echo "  - variance_vs_pcc_K300.png (variance vs PCC scatter plot)"
echo "============================================================================"
