import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Set

def read_GBM(data_path: Path) -> pd.DataFrame:
    """
    Read and process GBM (Glioblastoma) data from a text file.

    Parameters:
    -----------
    data_path : Path
        Path to the directory containing the GBM.txt file

    Returns:
    --------
    pd.DataFrame
        Processed GBM data with log2-transformed values and FPS_ prefixed sample IDs
    """

    gbm = pd.read_csv(data_path/'GBM.txt', sep='\t').transpose()
    gbm.columns = [i.split('_')[1] for i in gbm.columns]
    gbm = gbm.loc[:, gbm.std() != 0]

    #take the average of duplicate columns
    gbm = gbm.groupby(gbm.columns, axis=1).mean()

    gbm = gbm.reset_index()
    gbm['index'] = gbm['index'].apply(lambda x: 'FPS_' + x)
    gbm = gbm.set_index('index')

    #add one to all values and than take log2
    gbm = gbm.apply(lambda x: np.log2(x+1))

    return gbm

def read_PDAC(data_path: Path) -> pd.DataFrame:
    """
    Read and process PDAC (Pancreatic Ductal Adenocarcinoma) data from multiple files.

    Parameters:
    -----------
    data_path : Path
        Path to the directory containing PDAC related files (LMD_RNAseq files and HGNC mapping)

    Returns:
    --------
    pd.DataFrame
        Processed PDAC data with log2-transformed values and IEO_ prefixed sample IDs
    """

    hgncID_to_symbol = pd.read_csv(data_path/'hgncID_to_symbol.tsv', sep='\t')
    hgncID_to_symbol = pd.Series(hgncID_to_symbol['Approved symbol'].values, index=hgncID_to_symbol['HGNC ID']).to_dict()

    patiens_metadata = pd.read_csv(data_path/'LMD_RNAseq_annotation.txt', sep='\t')
    patients = set(patiens_metadata['Sample'].values)

    pdac = pd.read_csv(data_path/'LMD_RNAseq_raw.read.counts.txt', sep='\t')
    pdac = pdac[pdac['type_of_gene']=='protein-coding']
    pdac['Gene'] = pdac['HGNC'].map(hgncID_to_symbol)
    pdac = pdac.dropna(subset=['Gene'])
    pdac = pdac[['Gene']+list(patients)]
    pdac = pdac.set_index('Gene')
    pdac = pdac.transpose()

    #remove columns with 0 std
    pdac = pdac.loc[:, pdac.std() != 0]

    #take the average of duplicate columns
    pdac = pdac.groupby(pdac.columns, axis=1).mean()

    pdac = pdac.reset_index()

    pdac['index'] = pdac['index'].apply(lambda x: 'IEO_' + x)
    pdac = pdac.set_index('index')

    #add one to all values and than take log2
    pdac = pdac.apply(lambda x: np.log2(x+1))

    return pdac

def read_OSARC(data_path: Path) -> pd.DataFrame:
    """
    Read and process OSARC (Osteosarcoma) data from TPM file.

    Parameters:
    -----------
    data_path : Path
        Path to the directory containing the Osteo_tpm.tsv file

    Returns:
    --------
    pd.DataFrame
        Processed osteosarcoma data with log2-transformed values and OSARC_ prefixed sample IDs
    """

    osteo = pd.read_csv(data_path/'Osteo_tpm.tsv', sep='\t',index_col=1)
    osteo = osteo.drop(columns=['genes_id'],axis=1)
    osteo = osteo.transpose()

    #remove columns with 0 std
    osteo = osteo.loc[:, osteo.std() != 0]

    #take the average of duplicate columns
    osteo = osteo.groupby(osteo.columns, axis=1).mean()

    #add a OSTEO_ prefix to the indexes
    osteo.index = ['OSARC_' + str(i) for i in osteo.index]

    #add one to all values and than take log2
    osteo = osteo.apply(lambda x: np.log2(x+1))

    return osteo

def read_external(dataset: str, data_path: Path) -> pd.DataFrame:
    """
    Read and process external dataset based on the specified type.

    Parameters:
    -----------
    dataset : str
        Type of dataset to read ('GBM', 'PDAC', or 'OSARC')
    data_path : Path
        Path to the directory containing the dataset files

    Returns:
    --------
    pd.DataFrame
        Processed dataset with appropriate transformations

    Raises:
    -------
    ValueError
        If the specified dataset type is not found
    """

    if dataset == 'GBM':
        return read_GBM(data_path)
    
    elif dataset == 'PDAC':
        return read_PDAC(data_path)
    
    elif dataset == 'OSARC':
        return read_OSARC(data_path)
    
    else:
        raise ValueError('Dataset not found')

    