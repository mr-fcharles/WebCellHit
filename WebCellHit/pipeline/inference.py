from typing import Dict, Tuple, Optional, Union
import pandas as pd
import numpy as np
from pathlib import Path
import gc
from tqdm.auto import tqdm

import sys
sys.path.append('/home/fcarli/CellHit/')
from CellHit.models import EnsembleXGBoost
from CellHit.data import obtain_drugs_metadata
from CellHit.utils import QuantileScoreComputer

from .neighbors import FaissKNN, add_tcga_metadata, add_ccle_metadata
from .dataclasses import InferencePaths

def compute_model_predictions(
    tdf: pd.DataFrame,
    dataset: str,
    inference_paths: InferencePaths,
    drug_stats_path: Optional[Union[Path, str]] = None,
    drug_metadata_path: Optional[Union[Path, str]] = None,
    models_path: Optional[Union[Path, str]] = None
) -> pd.DataFrame:
    """
    Compute model predictions for input data using InferencePaths.
    
    Args:
        tdf: Input transcriptomic data
        dataset: Dataset name ('gdsc' or 'prism')
        inference_paths: InferencePaths object containing all necessary paths
        drug_stats_path: Path to drug statistics file (optional)
        drug_metadata_path: Path to drug metadata (optional)
    
    Returns:
        DataFrame with predictions
    """
    models_path = Path(models_path or inference_paths.pretrained_models_path)
    
    # Load drug stats and metadata
    drug_stats = pd.read_csv(drug_stats_path or inference_paths.drug_stats)
    median_mapper = dict(zip(drug_stats['Drug'], drug_stats['median']))
    mean_mapper = dict(zip(drug_stats['Drug'], drug_stats['mean']))
    std_mapper = dict(zip(drug_stats['Drug'], drug_stats['std']))
    
    drug_metadata = obtain_drugs_metadata(dataset, path=drug_metadata_path or inference_paths.drug_metadata)
    id_to_name_mapper = dict(zip(drug_metadata['DrugID'], drug_metadata['Drug']))
    id_to_repurposing_target_mapper = dict(zip(drug_metadata['DrugID'], drug_metadata['repurposing_target']))
    
    preds = []
    for model_path in tqdm(models_path.glob('*.xgb')):
    #for model_path in list(models_path.glob('*.xgb'))[:5]:
        stem = int(model_path.stem)
        name = id_to_name_mapper[stem]
        repurposing_target = id_to_repurposing_target_mapper[stem]
        
        model = EnsembleXGBoost.load_model(model_path)
        predictions = model.predict(tdf, return_shaps=True, return_stds=True)
        
        output = elaborate_output(
            predictions=predictions,
            model=model,
            data=tdf,
            mean_mapper=mean_mapper,
            std_mapper=std_mapper,
            drug_id=stem,
            drug_name=name,
            repurposing_target=repurposing_target
        )
        
        preds.append(output)
        del model
        gc.collect()
        
    return pd.concat(preds)

def compute_quantile_scores(
    preds: pd.DataFrame,
    inference_paths: InferencePaths,
    quantile_computer_path: Optional[Union[Path, str]] = None,
    add_new_cells: bool = True,
    n_jobs: int = 26
) -> pd.DataFrame:
    """
    Compute quantile scores for predictions using InferencePaths.
    
    Args:
        preds: Predictions DataFrame
        inference_paths: InferencePaths object containing all necessary paths
        quantile_computer_path: Path to quantile computer object (optional)
        add_new_cells: Whether to add new cells to the quantile computer (required when running inference on data different from CCLE or TCGA)
        n_jobs: Number of parallel jobs
        
    Returns:
        DataFrame with quantile scores added
    """
    quantile_computer = QuantileScoreComputer.load(
        quantile_computer_path or inference_paths.quantile_computer
    )

    if add_new_cells:
        quantile_computer.add_cells(preds,cell_col='index',score_col='prediction')
    
    preds['QuantileScore'] = quantile_computer.parallel_compute_score(
        preds, 
        cell_col='index',
        drug_col='DrugID',
        score_col='prediction',
        n_jobs=n_jobs
    )
    return preds

def compute_neighbors(
    data: pd.DataFrame,
    mode: str,
    inference_paths: InferencePaths,
    ccle_neighs_path: Optional[Union[Path, str]] = None,
    tcga_neighs_path: Optional[Union[Path, str]] = None,
    ccle_metadata_path: Optional[Union[Path, str]] = None,
    tcga_metadata_path: Optional[Union[Path, str]] = None
) -> pd.DataFrame:
    """
    Unified function to compute neighbors in either transcriptomic or response space.
    
    Args:
        data: Input data (transcriptomic data or predictions DataFrame)
        mode: Type of neighbors to compute ('transcriptomic' or 'response')
        inference_paths: InferencePaths object containing all necessary paths
        ccle_neighs_path: Path to CCLE neighbors (optional)
        tcga_neighs_path: Path to TCGA neighbors (optional)
        ccle_metadata_path: Path to CCLE metadata (optional)
        tcga_metadata_path: Path to TCGA metadata (optional)
    
    Returns:
        DataFrame with computed neighbors and annotations
    """
    # Prepare input data for response mode
    if mode == 'response':
        data = data[['index','DrugID','prediction']].pivot(
            index='index',
            columns='DrugID',
            values='prediction'
        )

    if ccle_neighs_path is not None:
        ccle_neighs = FaissKNN.from_file(ccle_neighs_path)
    else:
        ccle_neighs = FaissKNN.from_file(
            inference_paths.ccle_transcr_neighs if mode == 'transcriptomic' 
            else inference_paths.ccle_response_neighs
        )

    if tcga_neighs_path is not None:
        tcga_neighs = FaissKNN.from_file(tcga_neighs_path)
    else:
        tcga_neighs = FaissKNN.from_file(
            inference_paths.tcga_transcr_neighs if mode == 'transcriptomic' 
            else inference_paths.tcga_response_neighs
        )
    
    # Compute neighbors
    prefix = f"ccle_{mode}_"
    ccle_neighs_df = ccle_neighs.knn(data, k=1, skip_self=True)
    ccle_neighs_df.rename({'neighbour_point': f'{prefix}neigh'}, axis=1, inplace=True)
    
    prefix = f"tcga_{mode}_"
    tcga_neighs_df = tcga_neighs.knn(data, k=1, skip_self=True)
    tcga_neighs_df.rename({'neighbour_point': f'{prefix}neigh'}, axis=1, inplace=True)
    
    # Merge neighbors
    neighs_df = pd.merge(
        ccle_neighs_df[['query_point', f'ccle_{mode}_neigh']],
        tcga_neighs_df[['query_point', f'tcga_{mode}_neigh']],
        on='query_point',
        how='inner'
    )
    
    # Add annotations
    ccle_annot = add_ccle_metadata(
        neighs_df[f'ccle_{mode}_neigh'], 
        metadata_path=ccle_metadata_path or inference_paths.ccle_metadata
    )
    ccle_annot.columns = [f'ccle_{mode}_neigh', f'ccle_{mode}_neigh_name', f'ccle_{mode}_neigh_tissue']
    ccle_annot['query_point'] = neighs_df['query_point']
    ccle_annot.set_index('query_point', inplace=True)
    
    tcga_annot = add_tcga_metadata(
        neighs_df[f'tcga_{mode}_neigh'], 
        metadata_path=tcga_metadata_path or inference_paths.tcga_metadata
    )
    tcga_annot.columns = [f'tcga_{mode}_neigh', f'tcga_{mode}_neigh_tissue']
    tcga_annot['query_point'] = neighs_df['query_point']
    tcga_annot.set_index('query_point', inplace=True)
    
    # Merge annotations
    neighs_df = pd.concat([ccle_annot, tcga_annot], axis=1).reset_index()
    
    return neighs_df[[
        'query_point',
        f'ccle_{mode}_neigh',
        f'ccle_{mode}_neigh_name',
        f'ccle_{mode}_neigh_tissue',
        f'tcga_{mode}_neigh',
        f'tcga_{mode}_neigh_tissue'
    ]]

def run_full_inference(
    tdf: pd.DataFrame,
    dataset: str,
    inference_paths: InferencePaths,
    drug_stats_path: Optional[Union[Path, str]] = None,
    drug_metadata_path: Optional[Union[Path, str]] = None,
    models_path: Optional[Union[Path, str]] = None,
    quantile_computer_path: Optional[Union[Path, str]] = None,
    ccle_metadata_path: Optional[Union[Path, str]] = None,
    tcga_metadata_path: Optional[Union[Path, str]] = None,
    return_heatmap: bool = False,
    n_jobs: int = 26,
    add_new_cells: bool = True,
    **kwargs
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Run complete inference pipeline using InferencePaths.
    
    Args:
        tdf: Input transcriptomic data
        dataset: Dataset name
        inference_paths: InferencePaths object containing all necessary paths
        drug_stats_path: Path to drug statistics file (optional)
        drug_metadata_path: Path to drug metadata (optional)
        models_path: Path to models directory (optional)
        quantile_computer_path: Path to quantile computer object (optional)
        ccle_metadata_path: Path to CCLE metadata (optional)
        tcga_metadata_path: Path to TCGA metadata (optional)
        return_heatmap: Whether to return heatmap data
        n_jobs: Number of parallel jobs
        add_new_cells: Whether to add new cells to the quantile computer (required when running inference on data different from CCLE or TCGA)

    Returns:
        If return_heatmap is False, returns DataFrame with predictions and neighbors
        If return_heatmap is True, returns tuple of (predictions DataFrame, heatmap DataFrame)
    """
    # Compute transcriptomic neighbors
    transcr_neighs_df = compute_neighbors(tdf, 'transcriptomic', inference_paths, ccle_metadata_path, tcga_metadata_path, **kwargs)
    #compute predictions
    preds = compute_model_predictions(tdf, dataset, inference_paths, drug_stats_path, drug_metadata_path, models_path, **kwargs)
    #compute quantile scores
    preds = compute_quantile_scores(preds, inference_paths, quantile_computer_path, add_new_cells, n_jobs, **kwargs)
    #compute response neighbors
    response_neighs_df = compute_neighbors(preds, 'response', inference_paths, ccle_metadata_path, tcga_metadata_path, **kwargs)
    
    # Merge results
    final_df = pd.merge(preds, transcr_neighs_df, left_on='index', right_on='query_point', how='inner')
    final_df = pd.merge(final_df, response_neighs_df, left_on='index', right_on='query_point', how='inner')
    final_df.drop(columns=['query_point_x', 'query_point_y'], inplace=True)

    
    if return_heatmap:
        # Compute heatmap data
        drug_stats = pd.read_csv(drug_stats_path or inference_paths.drug_stats)
        median_mapper = dict(zip(drug_stats['Drug'], drug_stats['median']))
        
        heatmap_data = preds[['index','prediction','DrugName']].pivot(
            index='index',
            columns='DrugName',
            values='prediction'
        )
        for drug in heatmap_data.columns:
            heatmap_data[drug] = heatmap_data[drug] - median_mapper[drug]
            
        return final_df, heatmap_data
    
    return final_df


def elaborate_output(
    predictions: pd.DataFrame,
    model: EnsembleXGBoost,
    data: pd.DataFrame,
    drug_id: int,
    drug_name: str,
    mean_mapper: Dict[str, float],
    std_mapper: Dict[str, float],
    repurposing_target: str,
    source: Optional[str] = None,
    topk: int = 15
) -> pd.DataFrame:
    
    """
    Elaborate output of a prediction.

    Args:
        predictions: Predictions DataFrame
        model: Model object
        data: Input data
        drug_id: Drug ID
        drug_name: Drug name
        mean_mapper: Mean mapper
        std_mapper: Standard deviation mapper
        repurposing_target: Repurposing target
        source: Source (optional)
        topk: Topk (optional)
        
    Returns:
        DataFrame with elaborated output
    """

    gene_names = np.array(model.models[0].feature_names)

    if 'index' in data.columns:
        indexes = data['index'].values
    else:
        indexes = data.index.values

    preds = predictions['predictions']
    stds = predictions['std']

    #get shap values
    shaps = predictions['shap_values'].values

    #prediction dataframe
    preds_df = pd.DataFrame()
    
    if source is not None:
        preds_df['Source'] = source

    preds_df['DrugID'] = [drug_id]*len(indexes)
    preds_df['DrugName'] = [drug_name]*len(indexes)
    preds_df['index'] = indexes
    preds_df['prediction'] = (preds * std_mapper[drug_name]) + mean_mapper[drug_name]
    preds_df['std'] = stds * std_mapper[drug_name]

    #take topk 15 absolute values for each instance (most important features)
    topk_indexes = np.abs(shaps).argsort(axis=1)[:,-topk:]
    #transform the indexes into feature names
    topk_feature_names = gene_names[topk_indexes]
    #shap values
    values = shaps[np.arange(shaps.shape[0])[:,None], topk_indexes]
    #transform the bidimensional array into a list of lists
    topk_feature_names = topk_feature_names.tolist()

    #create a new column with the topk feature names
    preds_df['TopGenes'] = topk_feature_names
    preds_df['TopGenes'] = preds_df['TopGenes'].apply(lambda x: ','.join(x))
    preds_df['ShapDictionary'] = [dict(zip(topk_feature_names[i], values[i])) for i in range(values.shape[0])]
    preds_df['PutativeTarget'] = [repurposing_target]*len(indexes)

    return preds_df