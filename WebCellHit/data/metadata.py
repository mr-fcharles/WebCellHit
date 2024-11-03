import pandas as pd
import numpy as np
import faiss
import json

def calculate_tissue_distances(
    df: pd.DataFrame,
    selected_cell_lines: pd.DataFrame,
    tcga_oncotree_path: str,
    tissue_map_path: str,
    model_metadata_path: str,
    k: int = 10,
    return_summary_df: bool = True
) -> pd.DataFrame:
    """
    Calculate tissue-wise distances between selected CCLE and TCGA samples.
    
    Args:
        selected_cell_lines: DataFrame of selected cell lines from select_representative_cell_lines()
        df: DataFrame of celligner output
        tcga_oncotree_path: Path to TCGA oncotree data
        tissue_map_path: Path to tissue mapping pickle file
        model_metadata_path: Path to Model.csv containing cell line metadata
    
    Returns:
        DataFrame with tissue-wise distance statistics
    """
    # Load tissue mapping
    with open(tissue_map_path, 'r') as f:
        tissue_map = json.load(f)

    # Process TCGA and CCLE data
    tcga_oncontree = pd.read_csv(tcga_oncotree_path)
    metadata = pd.read_csv(model_metadata_path)[['ModelID', 'OncotreeCode']]
    
    # Map tissues
    tcga_oncontree['mainType'] = tcga_oncontree['oncotree_code'].map(tissue_map)
    metadata['mainType'] = metadata['OncotreeCode'].map(tissue_map)
    
    tcga_mapper = dict(zip(tcga_oncontree['sample_id'], tcga_oncontree['mainType']))
    ccle_mapper = dict(zip(metadata['ModelID'], metadata['mainType']))

    # Split and process TCGA and CCLE data
    tcga = df[df['Source'] == 'TCGA'].copy()
    ccle = df[df['Source'] == 'CCLE'].copy()

    tcga['oncotree_code'] = tcga.index.map(tcga_mapper)
    ccle['oncotree_code'] = ccle.index.map(ccle_mapper)
    
    tcga = tcga.dropna(subset=['oncotree_code'])
    ccle = ccle.dropna(subset=['oncotree_code'])
    ccle = ccle[ccle.index.isin(set(selected_cell_lines.index))]

    # Calculate distances per tissue
    tissue_dict = {}
    for tissue in ccle['oncotree_code'].unique():
        
        tdf_ccle = ccle[ccle['oncotree_code'] == tissue].drop(columns=['oncotree_code', 'Source'])
        tdf_tcga = tcga[tcga['oncotree_code'] == tissue].drop(columns=['oncotree_code', 'Source'])
        
        if len(tdf_ccle) > 0 and len(tdf_tcga) > 0:
            ccle_data = tdf_ccle.values.astype('float32')
            tcga_data = tdf_tcga.values.astype('float32')
            
            index = faiss.IndexFlatL2(ccle_data.shape[1])
            index.add(ccle_data)
            
            k = min(k, len(ccle_data))
            distances, _ = index.search(tcga_data, k)
            tissue_dict[tissue] = distances.ravel()

    
    if return_summary_df:
    
        # Calculate statistics
        stats = {
            'tissue': [],
            'mean_distance': [],
            'std_deviation': [],
            'min_distance': [],
            'max_distance': [],
            '25th_percentile': [],
            'median': [],
            '75th_percentile': [],
            'sample_count': []
        }

        for tissue, distances in tissue_dict.items():
            stats['tissue'].append(tissue)
            stats['mean_distance'].append(np.mean(distances))
            stats['std_deviation'].append(np.std(distances))
            stats['min_distance'].append(np.min(distances))
            stats['max_distance'].append(np.max(distances))
            stats['25th_percentile'].append(np.percentile(distances, 25))
            stats['median'].append(np.percentile(distances, 50))
            stats['75th_percentile'].append(np.percentile(distances, 75))
            stats['sample_count'].append(len(distances))

        return pd.DataFrame(stats)
    
    else:
        return tissue_dict
