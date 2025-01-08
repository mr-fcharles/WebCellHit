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

from .dataclasses import PreprocessPaths

from ..utils import tcgaCodeMap, ccleCodeMap
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
    preprocess_paths: PreprocessPaths,
    classifier_path: Optional[Union[Path, str]] = None,
    classifier_mapper_path: Optional[Union[Path, str]] = None
) -> pd.Series:
    """Classify tumor samples using pre-trained classifier
    
    Args:
        data: Input gene expression data
        classifier_path: Path to pre-trained classifier
        preprocess_paths: Paths to required models and data
    
    Returns:
        Series of classified tumor samples
    """
    classifier = xgb.Booster()
    classifier.load_model(classifier_path or str(preprocess_paths.classifier_path))
    
    with open(classifier_mapper_path or preprocess_paths.classifier_mapper_path) as f:
        code_mapper = json.load(f)
    reverse_code_mapper = {v: k for k, v in code_mapper.items()}
    
    genes = classifier.feature_names
    data_completed = data_frame_completer(data, genes)
    
    dmatrix = xgb.DMatrix(data_completed, enable_categorical=True)
    predictions = classifier.predict(dmatrix)
    return pd.Series(predictions, index=data_completed.index).map(reverse_code_mapper)


def batch_correct_ccle(
    data: pd.DataFrame,
    covariate_labels: Union[List[int], pd.Series],
    preprocess_paths: PreprocessPaths,
    ccle_data_path: Optional[Union[Path, str]] = None,
    ccle_metadata_path: Optional[Union[Path, str]] = None,
    ccle_code_map_path: Optional[Union[Path, str]] = None,
    batch_labels: Optional[List[int]] = None,
) -> pd.DataFrame:
    
    """
    Batch correct CCLE data

    Args:
        data: Input gene expression data
        covariate_labels: Covariate labels for correction
        preprocess_paths: Paths to required models and data
        ccle_data_path: Path to CCLE data
        ccle_metadata_path: Path to CCLE metadata
        batch_labels: List of batch labels, if not none, overrides batch correction (allows multiple batch correction in one go)
    Returns:
        Batch-corrected gene expression data
    """

    ccle_code_map = ccleCodeMap.load(ccle_code_map_path or preprocess_paths.ccle_code_map_path)
    
    ccle = pd.read_feather(ccle_data_path or preprocess_paths.ccle_data_path).set_index('index')
    ccle_metadata = pd.read_csv(ccle_metadata_path or preprocess_paths.ccle_metadata_path).sort_values(by='ModelID')
    ccle_metadata_mapper = dict(zip(ccle_metadata['ModelID'], ccle_metadata['OncotreeCode']))

    ccle['ccle_cancer_acronym'] = ccle.index.map(ccle_metadata_mapper).map(ccle_code_map.lookup_tissue)
    ccle = ccle.dropna(subset=['ccle_cancer_acronym'])

    ccle_covariates = ccle.pop('ccle_cancer_acronym')
    ccle_covariates = ccle_covariates.apply(ccle_code_map.lookup_ccle_code).to_list()

    # Prepare data for correction
    common_genes = list(set(data.columns).intersection(ccle.columns))
    data = data[common_genes].T
    ccle = ccle[common_genes].T

    overall_data = pd.concat([data, ccle], axis=1)
    if batch_labels is None:
        batch = [1] * data.shape[1] + [0] * ccle.shape[1]
    else:
        batch = batch_labels + [0] * ccle.shape[1] 
    
    covariate_labels = covariate_labels + ccle_covariates

    return overall_data, batch, covariate_labels

def batch_correct_tcga(
    data: pd.DataFrame,
    covariate_labels: Union[List[int], pd.Series],
    preprocess_paths: PreprocessPaths,
    tcga_data_path: Optional[Union[Path, str]] = None,
    tcga_metadata_path: Optional[Union[Path, str]] = None,
    tcga_code_map_path: Optional[Union[Path, str]] = None,
    batch_labels: Optional[List[int]] = None,
) -> pd.DataFrame:
    
    """
    Batch correct TCGA data
    
    Args:
        data: Input gene expression data
        covariate_labels: Covariate labels for correction
        preprocess_paths: Paths to required models and data
        tcga_data_path: Path to TCGA data
        tcga_metadata_path: Path to TCGA metadata
        tcga_code_map_path: Path to TCGA code map
        batch_labels: List of batch labels, if not none, overrides batch correction (allows multiple batch correction in one go)
        transform_source: Source of transformation ('target' or 'reference')
    Returns:
        Batch-corrected gene expression data
    """
    # Load TCGA reference data
    tcga = pd.read_feather(tcga_data_path or preprocess_paths.tcga_data_path).set_index('index')
    tcga_metadata = pd.read_csv(tcga_metadata_path or preprocess_paths.tcga_metadata_path)
    tcga_metadata_mapper = dict(zip(
        tcga_metadata['sample_id'],
        tcga_metadata['tcga_cancer_acronym']
    ))
    
    # Process TCGA data
    tcga['tcga_cancer_acronym'] = tcga.index.map(tcga_metadata_mapper)
    tcga = tcga.dropna(subset=['tcga_cancer_acronym'])

    #load code map
    tcga_code_map = tcgaCodeMap.load(tcga_code_map_path or preprocess_paths.tcga_code_map_path)
    tcga_covariates = tcga.pop('tcga_cancer_acronym')
    tcga_covariates = tcga_covariates.apply(tcga_code_map.lookup_tcga_code).to_list()
    #tcga = tcga.T
    
    # Prepare data for correction
    common_genes = list(set(data.columns).intersection(tcga.columns))
    data = data[common_genes].T
    tcga = tcga[common_genes].T
    
    # Combine data
    overall_data = pd.concat([data, tcga], axis=1)
    if batch_labels is None:
        batch = [1] * data.shape[1] + [0] * tcga.shape[1]
    else:
        batch = batch_labels + [0] * tcga.shape[1]
    
    covariate_labels = covariate_labels + tcga_covariates

    return overall_data, batch, covariate_labels
    
def batch_correct(
    data: pd.DataFrame,
    covariate_labels: Union[List[int], pd.Series],
    preprocess_paths: PreprocessPaths,
    tcga_data_path: Optional[Union[Path, str]] = None,
    tcga_metadata_path: Optional[Union[Path, str]] = None,
    tcga_code_map_path: Optional[Union[Path, str]] = None,
    ccle_data_path: Optional[Union[Path, str]] = None,
    ccle_metadata_path: Optional[Union[Path, str]] = None,
    ccle_code_map_path: Optional[Union[Path, str]] = None,
    batch_labels: Optional[List[int]] = None,
    transform_source: str = 'target'
) -> pd.DataFrame:
    """Perform batch correction using ComBat
    
    Args:
        data: Input gene expression data
        covariate_labels: Covariate labels for correction
        preprocess_paths: Paths to required models and data
        tcga_data_path: Path to TCGA data
        tcga_metadata_path: Path to TCGA metadata
        tcga_code_map_path: Path to TCGA code map
        ccle_data_path: Path to CCLE data
        ccle_metadata_path: Path to CCLE metadata
        ccle_code_map_path: Path to CCLE code map
        batch_labels: List of batch labels, if not none, overrides batch correction (allows multiple batch correction in one go)
        transform_source: Source of transformation ('target' or 'reference')
    Returns:
        Batch-corrected gene expression data
    """
    # Pre-process data based on transform source
    if transform_source == 'target':
        overall_data, batch, covariate_labels = batch_correct_tcga(
            data=data,
            covariate_labels=covariate_labels,
            preprocess_paths=preprocess_paths,
            tcga_data_path=tcga_data_path,
            tcga_metadata_path=tcga_metadata_path,
            tcga_code_map_path=tcga_code_map_path,
            batch_labels=batch_labels
        )
    else:
        overall_data, batch, covariate_labels = batch_correct_ccle(
            data=data,
            covariate_labels=covariate_labels,
            preprocess_paths=preprocess_paths,
            ccle_data_path=ccle_data_path,
            ccle_metadata_path=ccle_metadata_path,
            ccle_code_map_path=ccle_code_map_path,
            batch_labels=batch_labels
        )

    # Perform correction
    corrected = pycombat(overall_data, batch, covariate_labels).T
    return corrected.iloc[:data.shape[0]]

def impute_missing(
    data: pd.DataFrame,
    preprocess_paths: PreprocessPaths,
    covariate_labels: Union[List[int], pd.Series],
    imputer_path: Optional[Union[Path, str]] = None,
    tcga_code_map_path: Optional[Union[Path, str]] = None
) -> pd.DataFrame:
    """Impute missing values using pre-trained model
    
    Args:
        data: Input gene expression data
        preprocess_paths: Paths to required models and data
        covariate_labels: Covariate labels for correction
        imputer_path: Path to pre-trained imputer
        tcga_code_map_path: Path to TCGA code map

    Returns:
        Imputed gene expression data
    """
    model = xgb.Booster()
    model.load_model(imputer_path or str(preprocess_paths.imputer_path))
    
    #get missing genes
    genes = model.feature_names
    #data_frame_completer also guarantees that the order of the genes is the same as the order in the model
    data_completed, missing_genes, _ = data_frame_completer(data, genes, return_genes=True)
    #set the tumor type (used for inference)
    data_completed['tumor_y'] = covariate_labels

    #pass back to string since XGBoost expects strings
    tcga_code_map = tcgaCodeMap.load(tcga_code_map_path or preprocess_paths.tcga_code_map_path)
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
    preprocess_paths: PreprocessPaths,
    transform_source: str = 'target',
    device: str = 'cuda:0',
    celligner_path: Optional[Union[Path, str]] = None
) -> pd.DataFrame:
    """Transform data using Celligner
    
    Args:
        data: Input gene expression data
        preprocess_paths: Paths to required models and data
        transform_source: Source of transformation ('target' or 'reference')
        device: Device for Celligner transformation
        celligner_path: Path to Celligner model

    Returns:
        Transformed gene expression data
    """
    celligner = Celligner(device=device)
    celligner.load(celligner_path or preprocess_paths.celligner_path)
    return celligner.transform(data, compute_cPCs=False, transform_source=transform_source)

def preprocess_pipeline(
    data: pd.DataFrame,
    covariate_labels: Union[List[int], pd.Series],
    preprocess_paths: PreprocessPaths,
    map_umap: bool = False,
    device: str = 'cuda:0',
    classify: bool = True,
    transform_source: str = 'target',
    umap_path: Optional[Union[Path, str]] = None,
    batch_labels: Optional[List[int]] = None
) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
    """
    Complete preprocess pipeline for gene expression data
    
    Args:
        data: Input gene expression data
        covariate_labels: Covariate labels for correction
        preprocess_paths: Paths to required models and data
        map_umap: Whether to map data to UMAP space
        device: Device for Celligner transformation
        classify: Whether to classify tumor samples
        transform_source: Source of transformation ('target' or 'reference')
        umap_path: Path to UMAP model
        batch_labels: List of batch labels, if not none, overrides batch correction (allows multiple batch correction in one go)
    Returns:
        Dictionary of results
    """

    results = {}

    # Classification
    if classify:
        results['classification'] = classify_samples(data, preprocess_paths)
    
    # Batch correction
    corrected = batch_correct(data, covariate_labels, preprocess_paths, batch_labels=batch_labels,transform_source=transform_source)
    
    # Imputation
    imputed = impute_missing(corrected, preprocess_paths, covariate_labels)
    
    # Transformation
    transformed = celligner_transform_data(data=imputed, 
                                 preprocess_paths=preprocess_paths, 
                                 device=device, 
                                 transform_source=transform_source)
    
    umap_path = umap_path or preprocess_paths.umap_path
    
    if map_umap and umap_path:
        umap = ParametricUMAP.load(umap_path,device=device)
        embedding = umap.transform(transformed.values)
        
        umap_results = pd.DataFrame(
            embedding,
            columns=['UMAP1', 'UMAP2'],
            index=transformed.index
        )
        results['umap'] = umap_results

    results['transformed'] = transformed
    return results