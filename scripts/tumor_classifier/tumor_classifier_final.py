import os
import optuna
import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

import sys
sys.path.append(os.path.expanduser('~/WebCellHit'))
from WebCellHit.models.classifier import AutoXGBClassifier


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',type=str,default='~/WebCellHit/data/')
    parser.add_argument('--study_name',type=str,default='tumor_classifier')
    parser.add_argument('--database_path',type=str,default='~/WebCellHit/database/')
    parser.add_argument('--celligner_output_path',type=str,default='~/WebCellHit/data/transcriptomics/celligner_CCLE_TCGA_optimized.feather')
    parser.add_argument('--gpu_id',type=int,default=0)
    parser.add_argument('--output_path',type=str,default='~/WebCellHit/results/tumor_classifier/')
    args = parser.parse_args()

    #for each argument, expand the user path if needed
    for arg in ['data_path','database_path','celligner_output_path','output_path']:
        if '~' in getattr(args,arg):
            setattr(args,arg,Path(os.path.expanduser(getattr(args,arg))))

    #load the celligner output
    expression_data = pd.read_feather(args.celligner_output_path)
    expression_data = expression_data[expression_data['Source']=='TCGA'].drop(columns=['Source']).set_index('index')

    #read the tcga oncotree data
    tcga_oncotree_data = pd.read_csv(Path(args.data_path)/'metadata'/'tcga_oncotree_data.csv').dropna(subset=['tissue_type'])
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

    # Get number of classes
    num_classes = target_data['tissue_type'].nunique()

    # Split data into train and validation sets
    train_idx, val_idx = train_test_split(target_data.index, stratify=target_data['tissue_type'], test_size=0.15, random_state=42)

    X_train = expression_data.loc[train_idx]
    y_train = target_data.loc[train_idx, 'tissue_type']
    X_val = expression_data.loc[val_idx]
    y_val = target_data.loc[val_idx, 'tissue_type']

    # Load the study
    storage = f'sqlite:///{args.database_path}/{args.study_name}.db'
    study_name = args.study_name

    study = optuna.load_study(study_name=study_name, storage=storage)

    # Initialize the classifier
    classifier = AutoXGBClassifier(gpuID=args.gpu_id, num_classes=num_classes)

    # Get the best parameters
    best_params = study.best_params

    # Update the classifier with the best parameters
    classifier.best_params = best_params

    # Train the final model
    classifier.train_final_model(X_train, y_train, X_val, y_val)

    # Save the trained model
    classifier.save_model(args.output_path/'tumor_classifier_final_model.json')

    print("Final model trained and saved successfully.")