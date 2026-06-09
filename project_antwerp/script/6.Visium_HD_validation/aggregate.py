"""
aggregate.py
"""
def load_visium_hd_count_data(visium_path, gene_list, normalize=False):
    # Load Visium count matrix as AnnData
    adata = sc.read_10x_h5(f'{visium_path}/filtered_feature_bc_matrix.h5')
    adata.var_names_make_unique()
    xy = pd.read_parquet(f'{visium_path}/spatial/tissue_positions.parquet', engine='fastparquet')
    
    # Sort in same order as anndata object
    xy.set_index('barcode', inplace=True)
    xy = xy.loc[adata.obs.index]
    
    # Add scale factors
    adata.uns['spatial'] = {}
    scalefactors = json.load(open(f'{visium_path}/spatial/scalefactors_json.json', 'r'))
    adata.uns['spatial']['scalefactors'] = scalefactors

    # Add spatial coordinates to anndata object (fullres → hires 변환)
    scalef = scalefactors['tissue_hires_scalef']  # 0.09202595
    spatial = xy.loc[:, ['pxl_col_in_fullres', 'pxl_row_in_fullres']].to_numpy()
    spatial = spatial * scalef  # ← 이 한 줄만 추가됨
    adata.obsm['spatial'] = spatial

    # Normalize and log1p transformation
    if normalize:
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.scale(adata)

    # Load gene_list
    arr_gene_list = np.load(gene_list, allow_pickle=True)
    # Filter out unnecessary genes
    adata_subset = adata[:, adata.var_names.isin(arr_gene_list)]
    return adata_subset

def aggregate_adata(adata_pred, adata, pred_xy_coords):

    xy_coords = np.array(adata.obsm['spatial'])

    valid_bins = {}
    patch_size = sorted(np.unique(pred_xy_coords[:,0]))[1] - sorted(np.unique(pred_xy_coords[:,0]))[0]

    for i, coords in enumerate(pred_xy_coords):
        # Filter bins based on pixel coordinates
        x_coords = xy_coords[:,0]
        y_coords = xy_coords[:,1]
        valid_bin_ids = adata.obs_names[((x_coords >= coords[0]) & (x_coords < coords[0] + patch_size)) &
                                        ((y_coords >= coords[1]) & (y_coords < coords[1] + patch_size))]

        if len(valid_bin_ids) > 0:
            valid_bins[adata_pred.obs.index[i]] = np.array(valid_bin_ids)

    tile_ids = list(valid_bins.keys())
    agg_data = []

    for tile_id in tile_ids:
        bin_ids = valid_bins[tile_id]
        agg_data.append(np.sum(adata[bin_ids].X, axis=0))

    agg_data = np.concatenate(agg_data)

    # Define aggregated data into AnnData
    agg_adata = AnnData(X=agg_data)
    agg_adata.obs_names = tile_ids
    agg_adata.var_names = adata.var_names.values

    spatial = []
    for i in tile_ids:
        spatial.append(np.array(i.split('_')[-1].split(';'), dtype=int))

    slide_id = '_'.join(tile_ids[0].split('_')[:-1])
    spatial = np.array(spatial)

    agg_adata.obs['wsi'] = slide_id
    agg_adata.obs['xy_coords'] = [i.split('_')[-1].replace(';', '_') for i in tile_ids]
    agg_adata.obsm['spatial'] = spatial

    # Normalize
    sc.pp.normalize_total(agg_adata)
    sc.pp.log1p(agg_adata)

    # Scale data to unit variance and zero mean
    # sc.pp.scale(agg_adata)

    return agg_adata