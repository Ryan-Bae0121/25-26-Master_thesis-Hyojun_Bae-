#!/bin/bash
# check_samples_per_fold.sh

echo "Checking samples in each fold..."
echo "================================"

for fold in 01 02 03 04 05 06 07 08 09 10; do
    echo ""
    echo "Fold $fold:"
    
    csv_file="/project_antwerp/hbae/Loki_output/folds_10fold_hvg_predex/fold_${fold}_val.csv"
    
    if [ -f "$csv_file" ]; then
        # Extract unique sample IDs (GSM...)
        samples=$(awk -F'/' '{print $7}' "$csv_file" | grep "GSM" | sort | uniq)
        n_samples=$(echo "$samples" | wc -l)
        
        echo "  Samples: $n_samples"
        echo "$samples" | head -5
        
        if [ $n_samples -gt 5 ]; then
            echo "  ... (showing first 5)"
        fi
    else
        echo "  File not found: $csv_file"
    fi
done

echo ""
echo "================================"
echo "Summary: Check if any fold has multiple samples"