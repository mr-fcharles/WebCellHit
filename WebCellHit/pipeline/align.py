from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
import json
from combat.pycombat import pycombat
from typing import Union, Optional, Tuple, List
from dataclasses import dataclass
from parametric_umap import ParametricUMAP
from typing import Dict

#add celligner to path
import sys
sys.path.append('/home/fcarli/francisCelligner/celligner')
from celligner import Celligner

sys.path.append('/home/fcarli/WebCellHit/')
from WebCellHit.utils import tcgaCodeMap

@dataclass
class ModelPaths:
    """Container for model and data file paths
    
    Attributes:
        classifier_path: Path to classifier model
        classifier_mapper_path: Path to classifier mapper
        imputer_path: Path to imputer model
        celligner_path: Path to Celligner model
        tcga_data_path: Path to TCGA data
        tcga_metadata_path: Path to TCGA metadata
        tcga_code_map_path: Path to TCGA code map
        tcga_project_ids_path: Path to TCGA project ids
        umap_path: Path to UMAP model
    """
    classifier_path: Path
    classifier_mapper_path: Path 
    imputer_path: Path
    celligner_path: Path
    tcga_data_path: Path
    tcga_metadata_path: Path
    tcga_code_map_path: Path
    tcga_project_ids_path: Path
    umap_path: Optional[Path] = None

def data_frame_completer(
    df: pd.DataFrame, 
    genes: List[str],
    return_genes: bool = False,
    fill_value: float = np.nan
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, List[str], List[str]]]:
    """Complete dataframe with missing genes
    
    Args:
        df: Input dataframe
        genes: List of genes to complete dataframe with
        return_genes: Whether to return missing and common genes
        fill_value: Value to fill missing values with
    
    Returns:
        Completed dataframe, missing genes, common genes
    """
    missing_genes = list(set(genes) - set(df.columns))
    common_genes = list(set(df.columns).intersection(genes))
    df = df.reindex(columns=genes, fill_value=fill_value)
    if return_genes:
        return df, missing_genes, common_genes
    return df

def classify_samples(
    data: pd.DataFrame,
    model_paths: ModelPaths
) -> pd.Series:
    """Classify tumor samples using pre-trained classifier
    
    Args:
        data: Input gene expression data
        model_paths: Paths to required models and data
    
    Returns:
        Series of classified tumor samples
    """
    classifier = xgb.Booster()
    classifier.load_model(str(model_paths.classifier_path))
    
    with open(model_paths.classifier_mapper_path) as f:
        code_mapper = json.load(f)
    reverse_code_mapper = {v: k for k, v in code_mapper.items()}
    
    genes = classifier.feature_names
    data_completed = data_frame_completer(data, genes)
    
    dmatrix = xgb.DMatrix(data_completed, enable_categorical=True)
    predictions = classifier.predict(dmatrix)
    return pd.Series(predictions, index=data_completed.index).map(reverse_code_mapper)

def batch_correct(
    data: pd.DataFrame,
    covariate_labels: Union[List[int], pd.Series],
    model_paths: ModelPaths
) -> pd.DataFrame:
    """Perform batch correction using ComBat
    
    Args:
        data: Input gene expression data
        covariate_labels: Covariate labels for correction
        model_paths: Paths to required models and data
    
    Returns:
        Batch-corrected gene expression data
    """
    # Load TCGA reference data
    tcga = pd.read_feather(model_paths.tcga_data_path).set_index('index')
    tcga_metadata = pd.read_csv(model_paths.tcga_metadata_path)
    tcga_metadata_mapper = dict(zip(
        tcga_metadata['sample_id'],
        tcga_metadata['tcga_cancer_acronym']
    ))
    
    # Process TCGA data
    tcga['tcga_cancer_acronym'] = tcga.index.map(tcga_metadata_mapper)
    tcga = tcga.dropna(subset=['tcga_cancer_acronym'])

    #load code map
    tcga_code_map = tcgaCodeMap.load(model_paths.tcga_code_map_path)
    tcga_covariates = tcga.pop('tcga_cancer_acronym')
    tcga_covariates = tcga_covariates.apply(tcga_code_map.lookup_tcga_code).to_list()
    #tcga = tcga.T
    
    # Prepare data for correction
    common_genes = list(set(data.columns).intersection(tcga.columns))
    data = data[common_genes].T
    tcga = tcga[common_genes].T
    
    # Combine data
    overall_data = pd.concat([data, tcga], axis=1)
    batch = [0] * data.shape[1] + [1] * tcga.shape[1]
    covariate_labels = covariate_labels + tcga_covariates
    
    # Perform correction
    corrected = pycombat(overall_data, batch, covariate_labels).T
    return corrected.iloc[:data.shape[1]]

def impute_missing(
    data: pd.DataFrame,
    model_paths: ModelPaths,
    covariate_labels: Union[List[int], pd.Series]
) -> pd.DataFrame:
    """Impute missing values using pre-trained model
    
    Args:
        data: Input gene expression data
        model_paths: Paths to required models and data
        covariate_labels: Covariate labels for correction
    Returns:
        Imputed gene expression data
    """
    model = xgb.Booster()
    model.load_model(str(model_paths.imputer_path))
    
    #get missing genes
    genes = model.feature_names
    #data_frame_completer also guarantees that the order of the genes is the same as the order in the model
    data_completed, missing_genes, _ = data_frame_completer(data, genes, return_genes=True)
    #set the tumor type (used for inference)
    data_completed['tumor_y'] = covariate_labels

    #pass back to string since XGBoost expects strings
    tcga_code_map = tcgaCodeMap.load(model_paths.tcga_code_map_path)
    data_completed['tumor_y'] = data_completed['tumor_y'].apply(tcga_code_map.lookup_integer_code)
    #make tumor_y a categorical variable
    data_completed['tumor_y'] = pd.Categorical(data_completed['tumor_y'],
                                               categories=tcga_code_map.project_ids,
                                               ordered=False)
    
    #if there are missing genes, impute
    if missing_genes:
        inference = xgb.DMatrix(data_completed, enable_categorical=True)
        predictions = model.predict(inference)
        
        inferred = pd.DataFrame(
            predictions,
            columns=data_completed.drop(columns=['tumor_y']).columns,
            index=data_completed.index
        )
        
        #replace missing genes with inferred values
        for col in missing_genes:
            if col != 'tumor_y':
                data_completed[col] = inferred[col]
                
    return data_completed

def celligner_transform_data(
    data: pd.DataFrame,
    model_paths: ModelPaths,
    transform_source: str = 'target',
    device: str = 'cuda:0'
) -> pd.DataFrame:
    """Transform data using Celligner
    
    Args:
        data: Input gene expression data
        model_paths: Paths to required models and data
        transform_source: Source of transformation ('target' or 'reference')
        device: Device for Celligner transformation
    
    Returns:
        Transformed gene expression data
    """
    celligner = Celligner(device=device)
    celligner.load(model_paths.celligner_path)
    return celligner.transform(data, compute_cPCs=False, transform_source=transform_source)

def preprocess_pipeline(
    data: pd.DataFrame,
    covariate_labels: Union[List[int], pd.Series],
    model_paths: ModelPaths,
    map_umap: bool = False,
    device: str = 'cuda:0',
    classify: bool = True,
    transform_source: str = 'target'
) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
    """
    Complete preprocess pipeline for gene expression data
    
    Args:
        data: Input gene expression data
        covariate_labels: Covariate labels for correction
        model_paths: Paths to required models and data
        map_umap: Whether to map data to UMAP space
        device: Device for Celligner transformation
        classify: Whether to classify tumor samples
        transform_source: Source of transformation ('target' or 'reference')
    Returns:
        Dictionary of results
    """

    results = {}

    # Classification
    if classify:
        results['classification'] = classify_samples(data, model_paths)
    
    # Batch correction
    corrected = batch_correct(data, covariate_labels, model_paths)
    
    # Imputation
    imputed = impute_missing(corrected, model_paths, covariate_labels)
    
    # Transformation
    transformed = celligner_transform_data(data=imputed, 
                                 model_paths=model_paths, 
                                 device=device, 
                                 transform_source=transform_source)
    
    if map_umap and model_paths.umap_path:
        umap = ParametricUMAP.load(model_paths.umap_path)
        embedding = umap.transform(transformed.values)
        
        umap_results = pd.DataFrame(
            embedding,
            columns=['UMAP1', 'UMAP2'],
            index=transformed.index
        )
        results['umap'] = umap_results

    results['transformed'] = transformed
    return results