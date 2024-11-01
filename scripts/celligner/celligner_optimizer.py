import os
import re
import pickle
import optuna
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm.notebook import tqdm

import sys
celligner_path = Path(os.path.expanduser('~/francisCelligner/celligner/')) #change here with yours
sys.path.append(str(celligner_path))
sys.path.append(str(celligner_path / 'mnnpy'))
sys.path.append(os.path.expanduser('~/WebCellHit')) 

from celligner import Celligner
from WebCellHit.data import calculate_tissue_distances

def source_mapper(x,external=None):
        if x in set(ccle.index):
            return 'CCLE'
        elif x in set(tcga.index):
            return 'TCGA'
        else:
            return external


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data_path', type=str, default = '~/WebCellHit/data')
    parser.add_argument('--database_path', type=str, default = '~/WebCellHit/database')
    parser.add_argument('--n_trials', type=int, default=300, help='Number of trials')

    args = parser.parse_args()

    #expand user in all paths if ~ in path
    for arg in ['data_path','databases_path']:
        if '~' in args.__dict__[arg]:
            args.__dict__[arg] = args.__dict__[arg].replace('~',os.path.expanduser('~'))

    ###--CCLE--###
    data_path = Path(args.data_path)
    databases_path = Path(args.databases_path)
    #check if the data is already saved
    if not (data_path /'transcriptomics' / 'ccle_raw.feather').exists():

        #Read the CCLE data
        ccle = pd.read_csv(data_path/'transcriptomics'/'OmicsExpressionProteinCodingGenesTPMLogp1.csv',index_col=0)
        ccle.columns = [str(i).split(' ')[0] for i in ccle.columns]

        #compute the std of the data
        ccle_stds = ccle.apply(np.std,axis=0)
        #identify the genes with no variance and remove them
        ccle_stds = ccle_stds[ccle_stds>0]
        ccle_stds = set(ccle_stds.index)
        ccle = ccle[[i for i in ccle.columns if i in ccle_stds]]

        ###--TCGA--###
        tcga = pd.read_csv(data_path/'transcriptomics'/'TumorCompendium_v11_PolyA_hugo_log2tpm_58581genes_2020-04-09.tsv',sep='\t')
        tcga = tcga.set_index('Gene').transpose()

        #remove 0 std columns
        tcga_stds = tcga.apply(np.std,axis=0)
        tcga_stds = tcga_stds[tcga_stds>0]
        tcga_stds = set(tcga_stds.index)
        tcga = tcga[[i for i in tcga.columns if i in tcga_stds]]

        common_columns = set(ccle.columns).intersection(tcga.columns)
        ccle = ccle[list(common_columns)]
        tcga = tcga[list(common_columns)]

        tumor_samples = tcga

        #check if the data is already saved
        if not (data_path/'transcriptomics' / 'ccle_raw.feather').exists():
            ccle.reset_index().to_feather(data_path/'transcriptomics' / 'ccle_raw.feather')
        if not (data_path/'transcriptomics' / 'tcga_raw.feather').exists():
            tcga.reset_index().to_feather(data_path/'transcriptomics' / 'tcga_raw.feather')

    else:
        #load the data and set as index the first column
        ccle = pd.read_feather(data_path/'transcriptomics' / 'ccle_raw.feather').set_index('index')
        tcga = pd.read_feather(data_path/'transcriptomics' / 'tcga_raw.feather').set_index('index')
        tumor_samples = tcga

    #obtain selected cell lines to test the performance of the model
    #selected_cell_lines = select_representative_cell_lines(data_path/'metadata/Model.csv',data_path/'transcriptomics/OmicsExpressionProteinCodingGenesTPMLogp1.csv',data_path/'metadata/tissueMap.json')
    selected_cell_lines = pd.read_csv(data_path/'metadata/selected_cell_lines_50.csv',index_col=0)
    
    #create a optuna sampler
    sampler = optuna.samplers.TPESampler(multivariate=True,n_startup_trials=100)

    #create an optuna study
    study = optuna.create_study(direction='minimize',sampler=sampler,study_name='celligner_optimize',storage=f'sqlite:///{databases_path}/celligner_optimize.db',load_if_exists=True)

    for i in range(args.n_trials):

        trial = study.ask()

        # Suggest hyperparameters for trial
        top_k_genes = trial.suggest_categorical('top_k_genes', [i for i in range(500,2001,100)]) #from 500 to 2000 in steps of 100
        pca_ncomp = trial.suggest_categorical('pca_ncomp', [i for i in range(30,101,5)]) #from 30 to 100 in steps of 5
        snn_n_neighbors = trial.suggest_categorical('snn_n_neighbors', [i for i in range(10,51,5)]) #from 10 to 50 in steps of 5
        cpca_ncomp = trial.suggest_int('cpca_ncomp', 2, 8)
        alpha = trial.suggest_float('alpha', 0.1, 10.0, log=True)
        
        # Louvain parameters
        louvain_resolution = trial.suggest_float('louvain_resolution', 1.0, 10.0)
        louvain_params = {
            'resolution': louvain_resolution
        }
        
        # MNN parameters
        mnn_k1 = trial.suggest_int('mnn_k1', 3, 10)  # Neighbors of tumors in cell lines
        mnn_k2 = trial.suggest_categorical('mnn_k2', [i for i in range(20,101,10)])  #from 20 to 100 in steps of 10
        mnn_fk = trial.suggest_int('mnn_fk', 3, 10)
        mnn_cosine = trial.suggest_categorical('mnn_cosine_norm', [True, False])
        
        mnn_params = {
            'k1': mnn_k1,
            'k2': mnn_k2,
            'cosine_norm': mnn_cosine,
            'fk': mnn_fk
        }

        # Gather all parameters into a dictionary
        celligner_params = {
            'topKGenes': top_k_genes,
            'pca_ncomp': pca_ncomp, 
            'snn_n_neighbors': snn_n_neighbors,
            'cpca_ncomp': cpca_ncomp,
            'louvain_kwargs': louvain_params,
            'mnn_kwargs': mnn_params,
            'device': 'cuda:0',
            'alpha': alpha
        }

        #align the data
        my_alligner = Celligner(**celligner_params)
        my_alligner.fit(ccle,n_jobs=1)
        my_alligner.transform(tumor_samples)

        #save the data
        output = my_alligner.combined_output.copy()
        output['Source'] = output.index.map(lambda x: source_mapper(x))
        output = output[['Source'] + list(output.columns[:-1])]

        tissue_distances = calculate_tissue_distances(
            df=output,
            selected_cell_lines=selected_cell_lines,
            tcga_oncotree_path=data_path/'metadata/tcga_oncotree_data.csv',
            tissue_map_path=data_path/'metadata/tissueMap.json',
            model_metadata_path=data_path/'metadata/Model.csv'
        )

        study.tell(trial.number,tissue_distances['mean_distance'].mean())
