import pandas as pd
import sys
import pickle
import faiss
import pandas as pd
import numpy as np
import json
from pathlib import Path

class FaissKNN:
    """Class to perform KNN search using Faiss"""

    def __init__(self):
        self.index = None
        self.index_names = None

    def fit(self,df:pd.DataFrame):
        self.df = df
        self.build_faiss_index()
        
    def build_faiss_index(self):
        # Converting DataFrame to numpy array

        self.column_names = list(self.df.columns)
        
        self.index_names = list(self.df.index)
        feature_matrix = np.ascontiguousarray(self.df.to_numpy().astype('float32'))
        
        # Initializing and building the index
        self.index = faiss.IndexFlatL2(feature_matrix.shape[1])
        self.index.add(feature_matrix)
        
    def knn(self, query_df:pd.DataFrame, k:int, skip_self:bool=False):

        try:
            query_df = query_df[self.column_names]
        except:
            raise ValueError(f"query_df must have the same columns as the fitted df")
        
        query_names = list(query_df.index)
        query_matrix = np.ascontiguousarray(query_df.to_numpy().astype('float32'))
        
        # Search for k+1 neighbors if we want to skip the query point itself
        k_search = k + 1 if skip_self else k
        #k_search = k
        
        # Conduct the search
        distances, indices = self.index.search(query_matrix, k_search)

        #remove the query point itself if skip_self is True
        if skip_self:
            indices = indices[:,1:]
            distances = distances[:,1:]
        
        #print('prova')
        
        # Prepare the output DataFrame
        output_data = []
        for i, (dist_row, idx_row) in enumerate(zip(distances, indices)):
            for j, (dist, idx) in enumerate(zip(dist_row, idx_row)):
                output_data.append({
                    'query_point': query_names[i],
                    'neighbour_point': self.index_names[idx],
                    'ranking': j,
                    'distance': dist
                })
                
        return pd.DataFrame(output_data)
    
    def save(self, filepath:Path):
        """Save the fitted FaissKNN object to disk"""
        if self.index is None:
            raise ValueError("FaissKNN object must be fitted before saving")
            
        # Save the index to a temporary buffer
        index_buffer = faiss.serialize_index(self.index)
        
        # Save everything in a single file
        state = {
            'column_names': self.column_names,
            'index_buffer': index_buffer,
            'index_names': self.index_names,
            'df': self.df
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
            
    def load(self, filepath:Path):
        """Load a fitted FaissKNN object from disk"""
        # Load all attributes from the single file
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
            
        self.index = faiss.deserialize_index(state['index_buffer'])
        self.index_names = state['index_names']
        self.column_names = state['column_names']
        self.df = state['df']

    @classmethod
    def from_file(cls, filepath:Path):
        """Create a FaissKNN instance from saved files"""
        instance = cls()  # Create new instance
        
        # Load all attributes from the single file
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
            
        instance.index = faiss.deserialize_index(state['index_buffer'])
        instance.index_names = state['index_names']
        instance.df = state['df']
        instance.column_names = state['column_names']
        return instance


def compute_neighbors(query:pd.DataFrame, ccle_neighs_path:Path, tcga_neighs_path:Path, type='transcr'):
    """Find precomputednearest neighbors for query points in both CCLE and TCGA datasets.
    
    Args:
        query: Query points to find neighbors for
        ccle_neighs_path: Path to saved FaissKNN object for CCLE neighbors
        tcga_neighs_path: Path to saved FaissKNN object for TCGA neighbors 
        type: Type of neighbors to find, either 'transcr' (transcriptional) or 'response' (drug response)
        
    Returns:
        DataFrame containing nearest neighbors from both CCLE and TCGA for each query point,
        with columns ['Index', '{type}_CCLE_neigh', '{type}_TCGA_neigh']
    """
    #assert type is either 'transcr' or 'response'
    assert type in ['transcr','response'], 'type must be either transcr or response'

    ccle_neighs = FaissKNN.from_file(ccle_neighs_path)
    tcga_neighs = FaissKNN.from_file(tcga_neighs_path)

    #find neighs in ccle
    ccle_neighs_df = ccle_neighs.knn(query, k=1).rename(columns={'neighbour_point':f'{type}_CCLE_neigh','query_point':'Index'})[[f'{type}_CCLE_neigh','Index']]
    tcga_neighs_df = tcga_neighs.knn(query, k=1).rename(columns={'neighbour_point':f'{type}_TCGA_neigh','query_point':'Index'})[[f'{type}_TCGA_neigh','Index']]

    #merge the neighbors
    neighs = ccle_neighs_df.merge(tcga_neighs_df,on='Index')

    return neighs


def add_ccle_metadata(neighs:list,metadata_path:Path):
    """Add CCLE metadata to the neighbors DataFrame generated by compute_neighbors

    Args:
        neighs: list of neighbors in ACH format
        metadata_path: Path to the file containing the CCLE metadata (Model.csv)
    """

    #load the metadata
    metadata = pd.read_csv(metadata_path)

    out_df = pd.DataFrame()
    out_df['Index'] = neighs
    out_df['CellLineName'] = neighs.map(dict(zip(metadata['ModelID'],metadata['CellLineName'])))
    out_df['Tissue'] = neighs.map(dict(zip(metadata['ModelID'],metadata['OncotreeLineage'])))

    return out_df


def add_tcga_metadata(neighs:list,metadata_path:Path):
    """Add TCGA metadata to the neighbors DataFrame generated by compute_neighbors

    Args:
        neighs: list of neighbors in ACH format
        metadata_path: Path to the file containing the TCGA metadata (tcga_oncotree_data.csv)
    """
    #load the metadata
    metadata = pd.read_csv(metadata_path)

    out_df = pd.DataFrame()
    out_df['Index'] = neighs
    out_df['Tissue'] = neighs.map(dict(zip(metadata['sample_id'],metadata['tissue_type'])))

    return out_df