import sys

import pickle

from sklearn.metrics import mean_squared_error
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

sys.path.append('/home/fcarli/CellHit')
from CellHit.search_and_inference import search, inference
from CellHit.data import DatasetLoader,prepare_data

from sqlalchemy import create_engine
from AsyncDistribJobs.models import Job
from AsyncDistribJobs.operations import configure_database
from AsyncDistribJobs.operations import add_jobs, get_jobs_by_state, job_statistics,fetch_job
from AsyncDistribJobs.operations import process_job

def run_full_asynch_inference(args,**kwargs):

    inference_database_path = Path('/home/fcarli/WebCellHit/database/standalone.db')

    engine = create_engine(f'sqlite:///{inference_database_path}')
    configure_database(engine,reset=args["build_inference_db"])

    #if specified, build the search database
    if args["build_inference_db"]:

        #populate the search database
        loader = DatasetLoader(dataset=args['dataset'],
                                data_path=args['data_path'],
                                celligner_output_path=args['celligner_output_path'],
                                use_external_datasets=True,
                                samp_x_tissue=2,
                                random_state=0)
        
        drugs_ids = loader.get_drugs_ids()

        #if dataset is PRISM, inference only on drugs with corr > 0.2
        if args['dataset'] == 'prism':
            perfs = pd.read_csv(args['tabs_path']/'drugs_performances_PRISM.tsv',sep='\t')[['DrugID','MOA Corr']]
            drugs_ids = perfs[perfs['MOA Corr'] >= 0.2]['DrugID'].tolist()

            #TODO: REMEMBER TO REMOVE THIS
            #remove the ones already present in the direcoe
            processed = [i.stem for i in Path(results_path).iterdir() if i.suffix == '.csv']
            processed = set([int(i) for i in processed])

            drugs_ids = [i for i in drugs_ids if i not in processed]

        #for drugID in drugs_ids:
        #    add_job(payload={'drugID': int(drugID)},cid=f'{drugID}')

        jobs_list = [Job(state='pending',payload={'drugID': int(drugID),'dataset':args['dataset']},cid=f'{drugID}') for drugID in drugs_ids]        
        add_jobs(jobs_list)

    while len(get_jobs_by_state('pending')) > 0:
        process_job(process_drug,**args)



def process_drug(drugID, dataset, data_path,**kwargs):
    
    random_state = 0
    gene_selection_mode = 'moa_primed'
    use_external_datasets = False
    use_dumped_loaders = False
    data_path = Path('/home/fcarli/CellHit/data')

    #load the best params from previous search
    model_path = Path('/home/fcarli/CellHit/results/gdsc/search_and_inference/moa_primed/models') / f'{drugID}.pkl'

    with open(model_path, 'rb') as f:
        best_params = pickle.load(f)['best_params']

    results_dict = {}
    results_dict['drugID'] = drugID
    results_dict['MSEs'] = []
    results_dict['Corrs'] = []
    results_dict['Models'] = []

    for random_state in tqdm(range(20)):

        data_dict = prepare_data(drugID, dataset, random_state, 
                                gene_selection_mode, 
                                use_external_datasets=use_external_datasets,
                                data_path=data_path,celligner_output_path='/home/fcarli/francisCelligner/new_celligner_optimized.feather',
                                use_dumped_loaders=use_dumped_loaders)

        #join the two dictionaries
        inference_results = inference(best_params=best_params, refit=True, internal_inference=True, gene_selection_mode=gene_selection_mode, return_model=True,  **data_dict)
                
        results_dict['Models'].append(inference_results['model'])
        
        drug_mean = data_dict['loader'].get_drug_mean(drugID)
        drug_std = data_dict['loader'].get_drug_std(drugID)

        tdf = pd.DataFrame()
        tdf['DrugName'] = [data_dict['loader'].get_drug_name(drugID)]*len(inference_results['predictions'])
        tdf['DrugID'] = [drugID]*len(inference_results['predictions'])
        tdf['Predictions'] = inference_results['predictions']#.values
        tdf['Predictions_unscaled'] = (inference_results['predictions']* drug_std) + drug_mean
        tdf['Actual'] = data_dict['test_Y'].values
        tdf['Actual_unscaled'] = ((data_dict['test_Y'] * drug_std) + drug_mean).values
        tdf['DepMapID'] = data_dict['test_indexes']
        tdf['Seed'] = [random_state]*len(inference_results['predictions'])

        results_dict['MSEs'].append(mean_squared_error(tdf['Actual'],tdf['Predictions']))
        results_dict['Corrs'].append(np.corrcoef(tdf['Actual'],tdf['Predictions'])[0,1])
        
    with open(Path(f'/home/fcarli/WebCellHit/results/CellHit/models/{dataset}')/f'{drugID}.pkl','wb') as f:
        pickle.dump(results_dict,f)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Evaluate the cell tail effect')
    argparser.add_argument('--dataset', type=str, default='gdsc', help='Dataset to use')
    argparser.add_argument('--data_path', type=str, default='/home/fcarli/CellHit/data/', help='Path to the data')
    argparser.add_argument('--celligner_output_path', type=str, default='/home/fcarli/francisCelligner/new_celligner_optimized.feather', help='Path to the celligner output')
    #argparser.add_argument('--results_path', type=str, default='/home/fcarli/CellHit/results/', help='Path to the results')
    argparser.add_argument('--build_inference_db', default=False, action='store_true', help='Build the inference database')
    args = argparser.parse_args()

    data_path = Path(args.data_path)
    celligner_output_path = Path(args.celligner_output_path)

    args = vars(args)
    
    run_full_asynch_inference(args)

    