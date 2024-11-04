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
from WebCellHit.utils import read_external


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
    parser.add_argument('--use_external', action='store_true', help='Use external datasets')
    parser.add_argument('--database_path', type=str, default = '~/WebCellHit/database')
    parser.add_argument('--study_name', type=str, default='celligner_optimize_revised')
    parser.add_argument('--external_dataset', type=str, help='Name of the external dataset')
    parser.add_argument('--gpu_id', type=int, default=0)


    args = parser.parse_args()

    #expand user in all paths if ~ in path
    for arg in ['data_path','database_path']:
        if '~' in args.__dict__[arg]:
            args.__dict__[arg] = args.__dict__[arg].replace('~',os.path.expanduser('~'))

    ###--CCLE--###
    data_path = Path(args.data_path)

    #check if the data is already saved
    if not (data_path/'transcriptomics'/'ccle_raw.feather').exists():

        #Read the CCLE data
        ccle = pd.read_csv(data_path/'OmicsExpressionProteinCodingGenesTPMLogp1.csv',index_col=0)
        ccle.columns = [str(i).split(' ')[0] for i in ccle.columns]

        #compute the std of the data
        ccle_stds = ccle.apply(np.std,axis=0)
        #identify the genes with no variance and remove them
        ccle_stds = ccle_stds[ccle_stds>0]
        ccle_stds = set(ccle_stds.index)
        ccle = ccle[[i for i in ccle.columns if i in ccle_stds]]

        ###--TCGA--###
        tcga = pd.read_csv(data_path/'TumorCompendium_v11_PolyA_hugo_log2tpm_58581genes_2020-04-09.tsv',sep='\t')
        tcga = tcga.set_index('Gene').transpose()

        #remove 0 std columns
        tcga_stds = tcga.apply(np.std,axis=0)
        tcga_stds = tcga_stds[tcga_stds>0]
        tcga_stds = set(tcga_stds.index)
        tcga = tcga[[i for i in tcga.columns if i in tcga_stds]]

        #common columns
        if args.use_external:
            external = read_external(args.external_dataset,data_path)
            common_columns = set(ccle.columns).intersection(tcga.columns).intersection(external.columns)
            ccle = ccle[list(common_columns)]
            tcga = tcga[list(common_columns)]
            external = external[list(common_columns)]

            tumor_samples = pd.concat([tcga,external],axis=0)

        else:
            common_columns = set(ccle.columns).intersection(tcga.columns)
            ccle = ccle[list(common_columns)]
            tcga = tcga[list(common_columns)]

            tumor_samples = tcga

        #check if the data is already saved
        if not (data_path/'ccle_raw.feather').exists():
            ccle.reset_index().to_feather(data_path/'ccle_raw.feather')
        if not (data_path/'tcga_raw.feather').exists():
            tcga.reset_index().to_feather(data_path/'tcga_raw.feather')

    else:
        #load the data and set as index the first column
        ccle = pd.read_feather(data_path/'transcriptomics'/'ccle_raw.feather').set_index('index')
        tcga = pd.read_feather(data_path/'transcriptomics'/'tcga_raw.feather').set_index('index')
        tumor_samples = tcga

    study = optuna.load_study(study_name=args.study_name,storage=f'sqlite:///{args.database_path}/{args.study_name}.db')
    trials_df = study.trials_dataframe()
    trials_df['summary_value'] = trials_df[[i for i in trials_df.columns if 'value' in i]].mean(axis=1)
    trials_df = trials_df.sort_values(by='summary_value',ascending=True)
    best_trial_idx = int(trials_df['number'].values[0])
    trial = study.trials[best_trial_idx]
    best_params = trial.params

    #align the data
    my_alligner = Celligner(device=f'cuda:{args.gpu_id}',**best_params)
    my_alligner.fit(ccle,n_jobs=1)
    my_alligner.transform(tumor_samples)

    #save the data
    output = my_alligner.combined_output.copy()
    output['Source'] = output.index.map(lambda x: source_mapper(x,external=args.external_dataset))
    output = output[['Source'] + list(output.columns[:-1])]

    if args.use_external:
        suffix = 'CCLE_TCGA_' + args.external_dataset
    else:
        suffix = 'CCLE_TCGA'

    #save feather file and base alligner
    output.to_csv(data_path/'transcriptomics'/f'celligner_{suffix}_optimized_revised.csv')
    output.reset_index().to_feather(data_path/'transcriptomics'/f'celligner_{suffix}_optimized_revised.feather')
    my_alligner.save(data_path/'transcriptomics'/f'base_alligner_{suffix}_optimized_revised.pkl')
