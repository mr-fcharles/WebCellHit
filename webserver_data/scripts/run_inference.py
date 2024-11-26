import pandas as pd
import numpy as np
import argparse
import gc

from pathlib import Path

import sys
sys.path.append('/home/fcarli/CellHit/')
from CellHit.data import obtain_drugs_metadata
from CellHit.models import EnsembleXGBoost

sys.path.append('/home/fcarli/WebCellHit/webserver_data/')
from predictions import elaborate_output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',type=str,default='gdsc')
    args = parser.parse_args()

    #stats mappers
    drug_stats = pd.read_csv(f'/home/fcarli/WebCellHit/webserver_data/local_data/{args.dataset}_drug_stats.csv')
    median_mapper = dict(zip(drug_stats['Drug'],drug_stats['median']))
    mean_mapper = dict(zip(drug_stats['Drug'],drug_stats['mean']))
    std_mapper = dict(zip(drug_stats['Drug'],drug_stats['std']))

    #metadata mappers
    drug_metadata = obtain_drugs_metadata(args.dataset,path='/home/fcarli/WebCellHit/data/')
    id_to_name_mapper = dict(zip(drug_metadata['DrugID'],drug_metadata['Drug']))
    id_to_repurposing_target_mapper = dict(zip(drug_metadata['DrugID'],drug_metadata['repurposing_target']))

    #read all the data
    data = pd.read_feather('/home/fcarli/WebCellHit/data/transcriptomics/celligner_CCLE_TCGA_optimized_revised.feather').set_index('index')
    source = data.pop('Source').values

    #--- ACTUAL INFERENCE ---
    models_path = Path(f'/home/fcarli/WebCellHit/results/CellHit/inference_models/{args.dataset}')

    models = []
    drug_ids = []
    drug_names = []
    repurposing_targets = []

    #model loading
    for model_path in models_path.glob('*.xgb'):

        stem = int(model_path.stem)
        name = id_to_name_mapper[stem]
        repurposing_target = id_to_repurposing_target_mapper[stem]

        models.append(EnsembleXGBoost.load_model(model_path))
        drug_ids.append(stem)
        drug_names.append(name)
        repurposing_targets.append(repurposing_target)

    #model inference
    for idx,(model,drug_name,repurposing_target) in enumerate(zip(models,drug_names,repurposing_targets)):

        if Path(f'/home/fcarli/WebCellHit/webserver_data/local_data/inference_results/{args.dataset}/{drug_ids[idx]}.csv').exists():
            continue
    
        with EnsembleXGBoost.load_model(model_path) as model:
            predictions = model.predict(data,return_shaps=True,return_stds=True)
        output = elaborate_output(predictions=predictions,
                                    model=model,
                                    data=data,
                                    mean_mapper=mean_mapper,
                                    std_mapper=std_mapper,
                                    drug_id=drug_ids[idx],
                                    drug_name=drug_name,
                                    repurposing_target=repurposing_target,
                                    source=source)
        
        #save the output
        output.to_csv(f'/home/fcarli/WebCellHit/webserver_data/local_data/inference_results/{args.dataset}/{drug_ids[idx]}.csv')

        del predictions, output, model
        gc.collect()