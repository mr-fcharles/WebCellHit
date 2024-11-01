import argparse
import pandas as pd
import numpy as np
import faiss
import json
import os
from pathlib import Path

def select_representative_cell_lines(
    model_metadata_path: Path,
    expression_data_path: Path,
    tissue_map_path: Path,
    min_matching_percentage: float = 50.0,
    k: int = 10
) -> pd.DataFrame:
    """
    Select cell lines that have good tissue type matching with their neighbors.
    
    Args:
        model_metadata_path: Path to Model.csv containing cell line metadata
        expression_data_path: Path to expression data CSV
        tissue_map_path: Path to tissue mapping pickle file
        min_matching_percentage: Minimum percentage for cell line tissue matching
    
    Returns:
        DataFrame with selected cell lines and their tissue matching statistics
    """
    # Load tissue mapping
    with open(tissue_map_path, 'r') as f:
        tissue_map = json.load(f)

    # Load and process metadata
    metadata = pd.read_csv(model_metadata_path)[['ModelID', 'OncotreeCode']]
    metadata['Tissue'] = metadata['OncotreeCode'].map(tissue_map)
    metadata = metadata.dropna(subset=['Tissue'])
    tissue_mapper = dict(zip(metadata['ModelID'], metadata['Tissue']))

    # Load and process expression data
    ccle_expr = pd.read_csv(expression_data_path, index_col=0)
    ccle_expr.columns = [i.split(' ')[0] for i in ccle_expr.columns]
    ccle_expr = ccle_expr.loc[ccle_expr.index.isin(metadata['ModelID'])]

    # Calculate nearest neighbors
    data = ccle_expr.values.astype('float32')
    norms = np.linalg.norm(data, axis=1)
    normalized_data = data / norms[:, np.newaxis]

    dimension = data.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(normalized_data)

    distances, indices = index.search(normalized_data, k)

    # Create neighbor dataframe
    neighbor_df = pd.DataFrame(
        indices, 
        index=ccle_expr.index,
        columns=[f'neighbor_{i+1}' for i in range(k)]
    )

    # Map indices to cell line names
    for col in neighbor_df.columns:
        neighbor_df[col] = ccle_expr.index[neighbor_df[col]]
        neighbor_df[col] = neighbor_df[col].map(tissue_mapper)

    neighbor_df['original_tissue'] = neighbor_df.index.map(tissue_mapper)
    
    # Calculate matching percentages
    neighbor_df['matching_percentage'] = (
        neighbor_df.iloc[:, :-1].values == 
        neighbor_df['original_tissue'].values[:, None]
    ).mean(axis=1) * 100

    # Filter cell lines based on matching percentage
    return neighbor_df[neighbor_df['matching_percentage'] > min_matching_percentage]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_metadata_path',default='~/WebCellHit/data/metadata/Model.csv', type=str, help='Path to Model.csv containing cell line metadata')
    parser.add_argument('--expression_data_path',default='~/WebCellHit/data/transcriptomics/OmicsExpressionProteinCodingGenesTPMLogp1.csv', type=str, help='Path to expression data CSV')
    parser.add_argument('--tissue_map_path',default='~/WebCellHit/data/metadata/tissueMap.json', type=str, help='Path to tissue mapping pickle file')
    parser.add_argument('--output_path',default='~/WebCellHit/data/metadata/selected_cell_lines.csv', type=str, help='Path to output CSV file')
    parser.add_argument('--min_matching_percentage', type=float, default=50.0, help='Minimum percentage for cell line tissue matching')
    parser.add_argument('--k', type=int, default=10, help='Number of neighbors to consider')
    args = parser.parse_args()

    #expand user in all paths if ~ in path
    for arg in ['model_metadata_path','expression_data_path','tissue_map_path','output_path']:
        if '~' in args.__dict__[arg]:
            args.__dict__[arg] = args.__dict__[arg].replace('~',os.path.expanduser('~'))

    #modify selected cell lines name adding the min matching percentage
    args.output_path = args.output_path.replace('.csv',f'_{int(args.min_matching_percentage)}.csv')

    selected_cell_lines = select_representative_cell_lines(args.model_metadata_path, 
                                                           args.expression_data_path, 
                                                           args.tissue_map_path,
                                                           args.min_matching_percentage,
                                                           args.k)
    
    selected_cell_lines.to_csv(args.output_path, index=True)