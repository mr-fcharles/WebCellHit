import os
import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

import sys
sys.path.append(os.path.expanduser('~/WebCellHit'))
from WebCellHit.models.classifier import AutoXGBClassifier
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--database_path',type=str,default='~/WebCellHit/database/')
    parser.add_argument('--data_path',type=str,default='~/WebCellHit/data/')
    parser.add_argument('--study_name',type=str,default='tumor_classifier')
    parser.add_argument('--celligner_output_path',type=str,default='~/WebCellHit/data/transcriptomics/celligner_CCLE_TCGA_optimized.feather')
    parser.add_argument('--n_trials',type=int,default=300)
    parser.add_argument('--n_startup_trials',type=int,default=100)
    parser.add_argument('--optim_seed',type=int,default=0)
    parser.add_argument('--cv_splits',type=int,default=1)
    parser.add_argument('--gpu_id',type=int,default=0)
    
    args = parser.parse_args()
    database_path = f'sqlite:///{os.path.expanduser(args.database_path)}/{args.study_name}.db'
    study_name = args.study_name
    optim_seed = args.optim_seed
    cv_splits = args.cv_splits
    gpu_id = args.gpu_id
    n_trials = args.n_trials
    n_startup_trials = args.n_startup_trials
    
    data_path = Path(os.path.expanduser(args.data_path))

    #load the celligner output
    expression_data = pd.read_feather(os.path.expanduser(args.celligner_output_path))
    expression_data = expression_data[expression_data['Source']=='TCGA'].drop(columns=['Source']).set_index('index')

    #read the tcga oncotree data
    tcga_oncotree_data = pd.read_csv(data_path/'metadata'/'tcga_oncotree_data.csv').dropna(subset=['tissue_type'])
    tissue_mapper = dict(zip(tcga_oncotree_data['sample_id'],tcga_oncotree_data['tissue_type']))

    #target_data = expression_data.merge(tcga_oncotree_data,left_on='index',right_on='patient_id',how='inner').drop_duplicates()
    expression_data['tissue_type'] = expression_data.index.map(tissue_mapper)
    expression_data = expression_data.dropna(subset=['tissue_type'])

    #get the target data
    target_data = expression_data[['tissue_type']]

    #drop from expression data
    expression_data = expression_data.drop(columns=['tissue_type'])

    #transform project_id to categorical
    target_data['tissue_type'] = target_data['tissue_type'].astype('category').cat.codes

    #get number of classes
    num_classes = target_data['tissue_type'].nunique()

    #generate 3 cross validation splits
    cv_data = []

    for random_state in range(cv_splits):   

        train_idx,val_idx = train_test_split(target_data.index,stratify=target_data['tissue_type'],test_size=0.15,random_state=random_state)

        train_data = expression_data.loc[list(train_idx)]
        val_data = expression_data.loc[list(val_idx)]

        train_target = target_data.loc[list(train_idx)]
        val_target = target_data.loc[list(val_idx)]

        cv_data.append(((train_data,train_target),(val_data,val_target)))

    #initialize the classifier
    classifier = AutoXGBClassifier(gpuID=gpu_id,num_classes=num_classes)
    classifier.search(cv_data=cv_data,n_trials=n_trials,n_startup_trials=n_startup_trials,optim_seed=optim_seed,storage=database_path,study_name=study_name)
