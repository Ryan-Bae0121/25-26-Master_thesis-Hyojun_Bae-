#!/usr/bin/env python3
"""
save_embeddings.py
==================
fold당 한 번만 실행 - embedding 저장
이후 predict_fast.py로 빠르게 실험 가능

Usage:

    python save_embeddings.py \
        --train_csv fold_01_train.csv \
        --val_csv fold_01_val.csv \
        --hvg_file HVG_genelist.txt \
        --pretrained /path/to/checkpoint.pt \
        --output_dir /path/to/embeddings/fold_01 \
        --device cuda:0
python save_embeddings.py \
    --train_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/65um_fold_03/fold_03_train_65um.csv \
    --val_csv   /project_antwerp/hbae/Loki_output/10fold_csv_file/65um_fold_03/fold_03_val_65um.csv \
    --hvg_file  /project_antwerp/hbae/data/0317_hvg_2000_list.txt \
    --gt_expr   /project_antwerp/hbae/data/0317_HVG_NEW/combined_expression_matrix.npy \
    --gt_obs    /project_antwerp/hbae/data/0317_HVG_NEW/combined_obs.npy \
    --gene_list /project_antwerp/hbae/data/0317_HVG_NEW/all_shared_genes.txt \
    --pretrained /project_antwerp/hbae/Loki_output/65um_finetune_fold_03/finetune_65um_fold_03/checkpoints/epoch_latest.pt \
    --output_dir /project_antwerp/hbae/Loki_output/65um_finetune_embedding/fold_03 \
    --device cuda:0
    
fold02    
    nohup python save_embeddings.py \
    --train_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/HVG_10fold/fold_02_train.csv \
    --val_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/HVG_10fold/fold_02_val.csv \
    --hvg_file /project_antwerp/hbae/HVG_genelist.txt \
    --gt_obs /project_antwerp/hbae/data/combined_obs_fixed.npy \
    --pretrained /project_antwerp/hbae/Loki_output/finetune_10fold_runs_hvg_v2/fold_02/finetune_hvg_fold_02_20260213_200612/checkpoints/epoch_10.pt \
    --output_dir /project_antwerp/hbae/Loki_output/embeddings_finetuned/fold_02_fixed \
    --device cuda:0 > /tmp/fold02_fixed_log.txt 2>&1 &
진행상황
    tail -f /tmp/fold02_fixed_log.txt
수정버전 파인튜닝 f0ld 06
    nohup python save_embeddings.py \
    --train_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/0228_New_HVG_10fold/fold_06_train.csv \
    --val_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/0228_New_HVG_10fold/fold_06_val.csv \
    --hvg_file /project_antwerp/hbae/data/0228_HVG_NEW/0228_HVG_Finetune_gene_list_full.txt \
    --gt_obs /project_antwerp/hbae/data/0228_HVG_NEW/combined_obs.npy \
    --pretrained /project_antwerp/hbae/Loki_output/0228_finetune_10fold_runs_hvg_v2/fold_06/finetune_hvg_fold_06_20260302_203331/checkpoints/epoch_latest.pt \
    --output_dir /project_antwerp/hbae/Loki_output/0228_10fold_finetune_embedding/fold_06 \
    --device cuda:0 

fold 01
    python save_embeddings.py \
    --train_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/0228_New_HVG_10fold/fold_01_train.csv \
    --val_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/0228_New_HVG_10fold/fold_01_val.csv \
    --hvg_file /project_antwerp/hbae/data/0228_HVG_NEW/0228_HVG_Finetune_gene_list_full.txt \
    --gt_obs /project_antwerp/hbae/data/0228_HVG_NEW/combined_obs.npy \
    --pretrained /project_antwerp/hbae/Loki_output/0228_10epoch_finetune_10fold_runs_hvg_/fold_01/finetune_hvg_fold_01_20260303_223325/checkpoints/epoch_latest.pt \
    --output_dir /project_antwerp/hbae/Loki_output/0228_epoch10_finetune_embedding/fold_01 \
    --device cuda:0
> /tmp/fintune_fold01_fixed_log.txt 2>&1 &

fold 02
python save_embeddings.py \
    --train_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/0228_New_HVG_10fold/fold_02_train.csv \
    --val_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/0228_New_HVG_10fold/fold_02_val.csv \
    --hvg_file /project_antwerp/hbae/data/0228_HVG_NEW/0228_HVG_Finetune_gene_list_full.txt \
    --gt_obs /project_antwerp/hbae/data/0228_HVG_NEW/combined_obs.npy \
    --pretrained /project_antwerp/hbae/Loki_output/0228_10epoch_finetune_10fold_runs_hvg_/fold_02/finetune_hvg_fold_02_20260304_083726/checkpoints/epoch_latest.pt \
    --output_dir /project_antwerp/hbae/Loki_output/0228_epoch10_finetune_embedding/fold_02 \
    --device cuda:0

fold 03 
python save_embeddings.py \
    --train_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/0228_New_HVG_10fold/fold_03_train.csv \
    --val_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/0228_New_HVG_10fold/fold_03_val.csv \
    --hvg_file /project_antwerp/hbae/data/0228_HVG_NEW/0228_HVG_Finetune_gene_list_full.txt \
    --gt_obs /project_antwerp/hbae/data/0228_HVG_NEW/combined_obs.npy \
    --pretrained /project_antwerp/hbae/Loki_output/0228_10epoch_finetune_10fold_runs_hvg_/fold_03/finetune_hvg_fold_03_20260304_102238/checkpoints/epoch_latest.pt \
    --output_dir /project_antwerp/hbae/Loki_output/0228_epoch10_finetune_embedding/fold_03 \
    --device cuda:0

fold 04
python save_embeddings.py \
    --train_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/0228_New_HVG_10fold/fold_04_train.csv \
    --val_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/0228_New_HVG_10fold/fold_04_val.csv \
    --hvg_file /project_antwerp/hbae/data/0228_HVG_NEW/0228_HVG_Finetune_gene_list_full.txt \
    --gt_obs /project_antwerp/hbae/data/0228_HVG_NEW/combined_obs.npy \
    --pretrained /project_antwerp/hbae/Loki_output/0228_10epoch_finetune_10fold_runs_hvg_/fold_04/finetune_hvg_fold_04_20260304_192458/checkpoints/epoch_latest.pt \
    --output_dir /project_antwerp/hbae/Loki_output/0228_epoch10_finetune_embedding/fold_04 \
    --device cuda:0

fold 05
python save_embeddings.py \
    --train_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/0228_New_HVG_10fold/fold_05_train.csv \
    --val_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/0228_New_HVG_10fold/fold_05_val.csv \
    --hvg_file /project_antwerp/hbae/data/0228_HVG_NEW/0228_HVG_Finetune_gene_list_full.txt \
    --gt_obs /project_antwerp/hbae/data/0228_HVG_NEW/combined_obs.npy \
    --pretrained /project_antwerp/hbae/Loki_output/0228_10epoch_finetune_10fold_runs_hvg_/fold_05/finetune_hvg_fold_05_20260304_214350/checkpoints/epoch_latest.pt \
    --output_dir /project_antwerp/hbae/Loki_output/0228_epoch10_finetune_embedding/fold_05 \
    --device cuda:0

fold 06
python save_embeddings.py \
    --train_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/0228_New_HVG_10fold/fold_06_train.csv \
    --val_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/0228_New_HVG_10fold/fold_06_val.csv \
    --hvg_file /project_antwerp/hbae/data/0228_HVG_NEW/0228_HVG_Finetune_gene_list_full.txt \
    --gt_obs /project_antwerp/hbae/data/0228_HVG_NEW/combined_obs.npy \
    --pretrained /project_antwerp/hbae/Loki_output/0228_10epoch_finetune_10fold_runs_hvg_/fold_06/finetune_hvg_fold_06_20260304_232808/checkpoints/epoch_latest.pt \
    --output_dir /project_antwerp/hbae/Loki_output/0228_epoch10_finetune_embedding/fold_06 \
    --device cuda:0

fold 07
nohup python save_embeddings.py \
    --train_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/0228_New_HVG_10fold/fold_07_train.csv \
    --val_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/0228_New_HVG_10fold/fold_07_val.csv \
    --hvg_file /project_antwerp/hbae/data/0228_HVG_NEW/0228_HVG_Finetune_gene_list_full.txt \
    --gt_obs /project_antwerp/hbae/data/0228_HVG_NEW/combined_obs.npy \
    --pretrained /project_antwerp/hbae/Loki_output/0228_10epoch_finetune_10fold_runs_hvg_/fold_07/finetune_hvg_fold_07_20260305_011223/checkpoints/epoch_latest.pt \
    --output_dir /project_antwerp/hbae/Loki_output/0228_epoch10_finetune_embedding/fold_07 \
    --device cuda:0 > /tmp/fold07_fixed_log.txt 2>&1 &

    tail -f /tmp/fold07_fixed_log.txt
fold 08

python save_embeddings.py \
    --train_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/0228_New_HVG_10fold/fold_08_train.csv \
    --val_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/0228_New_HVG_10fold/fold_08_val.csv \
    --hvg_file /project_antwerp/hbae/data/0228_HVG_NEW/0228_HVG_Finetune_gene_list_full.txt \
    --gt_obs /project_antwerp/hbae/data/0228_HVG_NEW/combined_obs.npy \
    --pretrained /project_antwerp/hbae/Loki_output/0228_10epoch_finetune_10fold_runs_hvg_/fold_08/finetune_hvg_fold_08_20260305_025632/checkpoints/epoch_latest.pt \
    --output_dir /project_antwerp/hbae/Loki_output/0228_epoch10_finetune_embedding/fold_08 \
    --device cuda:0

fold 09

python save_embeddings.py \
    --train_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/0228_New_HVG_10fold/fold_09_train.csv \
    --val_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/0228_New_HVG_10fold/fold_09_val.csv \
    --hvg_file /project_antwerp/hbae/data/0228_HVG_NEW/0228_HVG_Finetune_gene_list_full.txt \
    --gt_obs /project_antwerp/hbae/data/0228_HVG_NEW/combined_obs.npy \
    --pretrained /project_antwerp/hbae/Loki_output/0228_10epoch_finetune_10fold_runs_hvg_/fold_09/finetune_hvg_fold_09_20260305_044043/checkpoints/epoch_latest.pt \
    --output_dir /project_antwerp/hbae/Loki_output/0228_epoch10_finetune_embedding/fold_09 \
    --device cuda:0

fold 10

python save_embeddings.py \
    --train_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/0228_New_HVG_10fold/fold_10_train.csv \
    --val_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/0228_New_HVG_10fold/fold_10_val.csv \
    --hvg_file /project_antwerp/hbae/data/0228_HVG_NEW/0228_HVG_Finetune_gene_list_full.txt \
    --gt_obs /project_antwerp/hbae/data/0228_HVG_NEW/combined_obs.npy \
    --pretrained /project_antwerp/hbae/Loki_output/0228_10epoch_finetune_10fold_runs_hvg_/fold_10/finetune_hvg_fold_10_20260305_062457/checkpoints/epoch_latest.pt \
    --output_dir /project_antwerp/hbae/Loki_output/0228_epoch10_finetune_embedding/fold_10 \
    --device cuda:0

0320 버전
fold 01
python save_embeddings.py \
    --train_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/0317_New_HVG_10fold/fold_01_train.csv \
    --val_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/0317_New_HVG_10fold/fold_01_val.csv \
    --hvg_file /project_antwerp/hbae/data/0317_training_data_excluding_GSE220978_and_19h1257/HVG_2000_genes.txt \
    --gt_obs /project_antwerp/hbae/data/0317_HVG_NEW/combined_obs.npy \
    --pretrained /project_antwerp/hbae/Loki_output/0317_10epoch_finetune_10fold_runs_hvg_/fold_01/finetune_hvg_fold_01_20260320_212457/checkpoints/epoch_latest.pt \
    --output_dir /project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new/fold_01 \
    --device cuda:0
> /tmp/fintune_fold01_fixed_log.txt 2>&1 &

python save_embeddings.py \
    --train_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/0317_New_HVG_10fold/fold_02_train.csv \
    --val_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/0317_New_HVG_10fold/fold_02_val.csv \
    --hvg_file /project_antwerp/hbae/data/0317_training_data_excluding_GSE220978_and_19h1257/HVG_2000_genes.txt \
    --gt_obs /project_antwerp/hbae/data/0317_HVG_NEW_norm/combined_obs.npy \
    --pretrained /project_antwerp/hbae/Loki_output/0317_10epoch_finetune_10fold_runs_hvg_/fold_02/finetune_hvg_fold_02_20260320_224414/checkpoints/epoch_latest.pt \
    --output_dir /project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding/fold_02 \
    --device cuda:0

new 버전
python save_embeddings.py \
    --train_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/0317_New_HVG_10fold/fold_01_train.csv \
    --val_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/0317_New_HVG_10fold/fold_01_val.csv \
    --hvg_file /project_antwerp/hbae/data/0317_hvg_2000_list.txt \
    --gt_expr /project_antwerp/hbae/data/0317_HVG_NEW/combined_expression_matrix.npy \
    --gt_obs  /project_antwerp/hbae/data/0317_HVG_NEW/combined_obs.npy \
    --gene_list /project_antwerp/hbae/data/0317_HVG_NEW/all_shared_genes.txt \
    --pretrained /project_antwerp/hbae/Loki_output/0317_10epoch_finetune_10fold_runs_hvg_/fold_01/finetune_hvg_fold_01_20260320_212457/checkpoints/epoch_latest.pt \
    --output_dir /project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new/fold_01 \
    --device cuda:0

python save_embeddings.py \
    --train_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/0317_New_HVG_10fold/fold_02_train.csv \
    --val_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/0317_New_HVG_10fold/fold_02_val.csv \
    --hvg_file /project_antwerp/hbae/data/0317_hvg_2000_list.txt \
    --gt_expr /project_antwerp/hbae/data/0317_HVG_NEW/combined_expression_matrix.npy \
    --gt_obs  /project_antwerp/hbae/data/0317_HVG_NEW/combined_obs.npy \
    --gene_list /project_antwerp/hbae/data/0317_HVG_NEW/all_shared_genes.txt \
    --pretrained /project_antwerp/hbae/Loki_output/0317_10epoch_finetune_10fold_runs_hvg_/fold_02/finetune_hvg_fold_02_20260320_224414/checkpoints/epoch_latest.pt \
    --output_dir /project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new/fold_02 \
    --device cuda:0

python save_embeddings.py \
    --train_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/0317_New_HVG_10fold/fold_03_train.csv \
    --val_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/0317_New_HVG_10fold/fold_03_val.csv \
    --hvg_file /project_antwerp/hbae/data/0317_hvg_2000_list.txt \
    --gt_expr /project_antwerp/hbae/data/0317_HVG_NEW/combined_expression_matrix.npy \
    --gt_obs  /project_antwerp/hbae/data/0317_HVG_NEW/combined_obs.npy \
    --gene_list /project_antwerp/hbae/data/0317_HVG_NEW/all_shared_genes.txt \
    --pretrained /project_antwerp/hbae/Loki_output/0317_10epoch_finetune_10fold_runs_hvg_/fold_03/finetune_hvg_fold_03_20260321_000327/checkpoints/epoch_latest.pt \
    --output_dir /project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new/fold_03 \
    --device cuda:0

python save_embeddings.py \
    --train_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/0317_New_HVG_10fold/fold_04_train.csv \
    --val_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/0317_New_HVG_10fold/fold_04_val.csv \
    --hvg_file /project_antwerp/hbae/data/0317_hvg_2000_list.txt \
    --gt_expr /project_antwerp/hbae/data/0317_HVG_NEW/combined_expression_matrix.npy \
    --gt_obs  /project_antwerp/hbae/data/0317_HVG_NEW/combined_obs.npy \
    --gene_list /project_antwerp/hbae/data/0317_HVG_NEW/all_shared_genes.txt \
    --pretrained /project_antwerp/hbae/Loki_output/0317_10epoch_finetune_10fold_runs_hvg_/fold_04/finetune_hvg_fold_04_20260323_194045/checkpoints/epoch_latest.pt \
    --output_dir /project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new/fold_04 \
    --device cuda:0

python save_embeddings.py \
    --train_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/0317_New_HVG_10fold/fold_05_train.csv \
    --val_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/0317_New_HVG_10fold/fold_05_val.csv \
    --hvg_file /project_antwerp/hbae/data/0317_hvg_2000_list.txt \
    --gt_expr /project_antwerp/hbae/data/0317_HVG_NEW/combined_expression_matrix.npy \
    --gt_obs  /project_antwerp/hbae/data/0317_HVG_NEW/combined_obs.npy \
    --gene_list /project_antwerp/hbae/data/0317_HVG_NEW/all_shared_genes.txt \
    --pretrained /project_antwerp/hbae/Loki_output/0317_10epoch_finetune_10fold_runs_hvg_/fold_05/finetune_hvg_fold_05_20260323_205941/checkpoints/epoch_latest.pt \
    --output_dir /project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new/fold_05 \
    --device cuda:0
    
python save_embeddings.py \
    --train_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/0317_New_HVG_10fold/fold_06_train.csv \
    --val_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/0317_New_HVG_10fold/fold_06_val.csv \
    --hvg_file /project_antwerp/hbae/data/0317_hvg_2000_list.txt \
    --gt_expr /project_antwerp/hbae/data/0317_HVG_NEW/combined_expression_matrix.npy \
    --gt_obs  /project_antwerp/hbae/data/0317_HVG_NEW/combined_obs.npy \
    --gene_list /project_antwerp/hbae/data/0317_HVG_NEW/all_shared_genes.txt \
    --pretrained /project_antwerp/hbae/Loki_output/0317_10epoch_finetune_10fold_runs_hvg_/fold_06/finetune_hvg_fold_06_20260323_221515/checkpoints/epoch_latest.pt \
    --output_dir /project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new/fold_06 \
    --device cuda:0

python save_embeddings.py \
    --train_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/0317_New_HVG_10fold/fold_07_train.csv \
    --val_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/0317_New_HVG_10fold/fold_07_val.csv \
    --hvg_file /project_antwerp/hbae/data/0317_hvg_2000_list.txt \
    --gt_expr /project_antwerp/hbae/data/0317_HVG_NEW/combined_expression_matrix.npy \
    --gt_obs  /project_antwerp/hbae/data/0317_HVG_NEW/combined_obs.npy \
    --gene_list /project_antwerp/hbae/data/0317_HVG_NEW/all_shared_genes.txt \
    --pretrained /project_antwerp/hbae/Loki_output/0317_10epoch_finetune_10fold_runs_hvg_/fold_07/finetune_hvg_fold_07_20260324_095306/checkpoints/epoch_latest.pt \
    --output_dir /project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new/fold_07 \
    --device cuda:0

python save_embeddings.py \
    --train_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/0317_New_HVG_10fold/fold_08_train.csv \
    --val_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/0317_New_HVG_10fold/fold_08_val.csv \
    --hvg_file /project_antwerp/hbae/data/0317_hvg_2000_list.txt \
    --gt_expr /project_antwerp/hbae/data/0317_HVG_NEW/combined_expression_matrix.npy \
    --gt_obs  /project_antwerp/hbae/data/0317_HVG_NEW/combined_obs.npy \
    --gene_list /project_antwerp/hbae/data/0317_HVG_NEW/all_shared_genes.txt \
    --pretrained /project_antwerp/hbae/Loki_output/0317_10epoch_finetune_10fold_runs_hvg_/fold_08/finetune_hvg_fold_08_20260324_144021/checkpoints/epoch_latest.pt \
    --output_dir /project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new/fold_08 \
    --device cuda:0

python save_embeddings.py \
    --train_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/0317_New_HVG_10fold/fold_09_train.csv \
    --val_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/0317_New_HVG_10fold/fold_09_val.csv \
    --hvg_file /project_antwerp/hbae/data/0317_hvg_2000_list.txt \
    --gt_expr /project_antwerp/hbae/data/0317_HVG_NEW/combined_expression_matrix.npy \
    --gt_obs  /project_antwerp/hbae/data/0317_HVG_NEW/combined_obs.npy \
    --gene_list /project_antwerp/hbae/data/0317_HVG_NEW/all_shared_genes.txt \
    --pretrained /project_antwerp/hbae/Loki_output/0317_10epoch_finetune_10fold_runs_hvg_/fold_09/finetune_hvg_fold_09_20260324_163351/checkpoints/epoch_latest.pt \
    --output_dir /project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new/fold_09 \
    --device cuda:0

python save_embeddings.py \
    --train_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/0317_New_HVG_10fold/fold_10_train.csv \
    --val_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/0317_New_HVG_10fold/fold_10_val.csv \
    --hvg_file /project_antwerp/hbae/data/0317_hvg_2000_list.txt \
    --gt_expr /project_antwerp/hbae/data/0317_HVG_NEW/combined_expression_matrix.npy \
    --gt_obs  /project_antwerp/hbae/data/0317_HVG_NEW/combined_obs.npy \
    --gene_list /project_antwerp/hbae/data/0317_HVG_NEW/all_shared_genes.txt \
    --pretrained /project_antwerp/hbae/Loki_output/0317_10epoch_finetune_10fold_runs_hvg_/fold_10/finetune_hvg_fold_10_20260324_183036/checkpoints/epoch_latest.pt \
    --output_dir /project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_new/fold_10 \
    --device cuda:0
    
norm 버전
python save_embeddings.py \
    --train_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/0317_New_HVG_10fold/fold_01_train.csv \
    --val_csv /project_antwerp/hbae/Loki_output/10fold_csv_file/0317_New_HVG_10fold/fold_01_val.csv \
    --hvg_file /project_antwerp/hbae/data/0317_training_data_excluding_GSE220978_and_19h1257/HVG_2000_genes.txt \
    --gt_expr /project_antwerp/hbae/data/0317_HVG_NEW_norm/combined_expression_matrix.npy \
    --gt_obs  /project_antwerp/hbae/data/0317_HVG_NEW_norm/combined_obs.npy \
    --gene_list /project_antwerp/hbae/data/0317_HVG_NEW_norm/31s_all_shared_genes.txt \
    --pretrained /project_antwerp/hbae/Loki_output/0317_10epoch_finetune_10fold_runs_hvg_/fold_01/finetune_hvg_fold_01_20260320_212457/checkpoints/epoch_latest.pt \
    --output_dir /project_antwerp/hbae/Loki_output/0317_epoch10_finetune_embedding_norm/fold_01 \
    --device cuda:0
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


# def load_omiclip(checkpoint_path, device):
  #  from open_clip import create_model_from_pretrained, get_tokenizer
   # model, preprocess = create_model_from_pretrained('coca_ViT-L-14', pretrained=checkpoint_path, device=device,weights_only=False) 
        #zero shot일떄만 weights_only=False 추가 , finetuning이면 weights_only=False 이거 제외)
    #model.eval()
    #return model, preprocess
def load_omiclip(checkpoint_path, device):
    import open_clip
    
    model, _, preprocess = open_clip.create_model_and_transforms(
        'coca_ViT-L-14', pretrained=None)
    
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # 파인튜닝 체크포인트인지 확인 후 분기
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']   # ← 파인튜닝 체크포인트
    else:
        state_dict = ckpt.get('state_dict', ckpt)  # ← 원본 pretrained
    
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    return model, preprocess


@torch.no_grad()
def encode_images_batch(model, image_paths, preprocess, device, batch_size=64):
    all_embs = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Encoding images"):
        batch_paths = image_paths[i:i+batch_size]
        images = []
        for path in batch_paths:
            try:
                img = Image.open(path).convert('RGB')
                images.append(preprocess(img))
            except:
                images.append(torch.zeros(3, 224, 224))
        img_tensor = torch.stack(images).to(device)
        emb = model.encode_image(img_tensor)
        if isinstance(emb, tuple):
            emb = emb[0]
        emb = F.normalize(emb, dim=-1)
        all_embs.append(emb.cpu())
    return torch.cat(all_embs, dim=0)


@torch.no_grad()
def encode_gene_sentences(model, gene_lists, tokenizer, device, batch_size=256):
    all_embs = []
    for i in tqdm(range(0, len(gene_lists), batch_size), desc="Encoding genes"):
        batch = gene_lists[i:i+batch_size]
        sentences = [' '.join(genes) for genes in batch]
        tokens = tokenizer(sentences).to(device)
        emb = model.encode_text(tokens)
        emb = F.normalize(emb, dim=-1)
        all_embs.append(emb.cpu())
    return torch.cat(all_embs, dim=0)


def main(args):
    import pandas as pd
    import open_clip

    device = torch.device(args.device)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Load model
    print("[1] Loading OmiCLIP...")
    model, preprocess = load_omiclip(args.pretrained, device)
    tokenizer = open_clip.get_tokenizer('coca_ViT-L-14')

    # 2. Load GT
    print("[2] Loading GT...")
    gt_expr = np.load(args.gt_expr)
    gt_obs  = np.load(args.gt_obs, allow_pickle=True)
    all_genes = open(args.gene_list).read().strip().split('\n')
    obs_to_idx = {b: i for i, b in enumerate(gt_obs)}
    gene_to_idx = {g: i for i, g in enumerate(all_genes)}

    hvg_genes = open(args.hvg_file).read().strip().split('\n')
    hvg_indices = [gene_to_idx[g] for g in hvg_genes if g in gene_to_idx]
    print(f"  HVG: {len(hvg_indices)}")

    # 3. Load CSVs
    def extract_obs_key(filepath):
        parts = filepath.split('/')
        patches_idx = parts.index('patches')
        sample_id = parts[patches_idx - 1]
        barcode = parts[-1].replace('.png', '')
        return f'{sample_id}_{barcode}_hires'

    train_df = pd.read_csv(args.train_csv)
    val_df   = pd.read_csv(args.val_csv)
    if 'obs_key' not in train_df.columns:
        train_df['obs_key'] = train_df['img_path'].apply(extract_obs_key)
    if 'obs_key' not in val_df.columns:
        val_df['obs_key']   = val_df['img_path'].apply(extract_obs_key)

    train_df = train_df[train_df['obs_key'].isin(obs_to_idx)].reset_index(drop=True)
    val_df   = val_df[val_df['obs_key'].isin(obs_to_idx)].reset_index(drop=True)
    print(f"  Train: {len(train_df):,} / Val: {len(val_df):,}")

    # 4. Encode train
    print("[3] Encoding train...")
    train_img_embs = encode_images_batch(model, train_df['img_path'].tolist(), preprocess, device)

    train_gene_lists, train_exprs_list = [], []
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Preparing train"):
        train_gene_lists.append(row['label'].split())
        spot_idx = obs_to_idx[row['obs_key']]
        train_exprs_list.append(gt_expr[spot_idx, hvg_indices])

    train_text_embs = encode_gene_sentences(model, train_gene_lists, tokenizer, device)
    train_exprs = np.array(train_exprs_list)

    # 5. Encode val
    print("[4] Encoding val...")
    val_img_embs = encode_images_batch(model, val_df['img_path'].tolist(), preprocess, device)

    val_exprs_list = []
    for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Preparing val"):
        spot_idx = obs_to_idx[row['obs_key']]
        val_exprs_list.append(gt_expr[spot_idx, hvg_indices])
    val_exprs = np.array(val_exprs_list)

    # 6. Save
    print("[5] Saving embeddings...")
    np.save(out / 'train_img_embs.npy',  train_img_embs.numpy())
    np.save(out / 'train_text_embs.npy', train_text_embs.numpy())
    np.save(out / 'train_exprs.npy',     train_exprs)
    np.save(out / 'val_img_embs.npy',    val_img_embs.numpy())
    np.save(out / 'val_exprs.npy',       val_exprs)

    print(f"\n✅ Saved to {out}")
    print(f"  train_img_embs:  {train_img_embs.shape}")
    print(f"  train_text_embs: {train_text_embs.shape}")
    print(f"  train_exprs:     {train_exprs.shape}")
    print(f"  val_img_embs:    {val_img_embs.shape}")
    print(f"  val_exprs:       {val_exprs.shape}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv",  required=True)
    p.add_argument("--val_csv",    required=True)
    p.add_argument("--hvg_file",   required=True)
    p.add_argument("--gt_expr",    default="/project_antwerp/hbae/data/0228_HVG_NEW/combined_expression_matrix.npy")
    p.add_argument("--gt_obs",     default="/project_antwerp/hbae/data/0228_HVG_NEW/combined_obs.npy")
    p.add_argument("--gene_list",  default="/project_antwerp/hbae/data/0228_HVG_NEW/ST_36s_all_shared_genes.txt")
    p.add_argument("--pretrained", default="/project_antwerp/assets/loki_ckpts/checkpoint.pt")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device",     default="cuda:0")
    args = p.parse_args()
    main(args)