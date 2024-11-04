import sys
sys.path.append('/home/fcarli/CellHit')

import pandas as pd

from pathlib import Path

from CellHit.search_and_inference import inference
from CellHit.data import DatasetLoader, prepare_data

from sqlalchemy import create_engine
from AsyncDistribJobs.models import Job
from AsyncDistribJobs.operations import configure_database
from AsyncDistribJobs.operations import add_jobs, get_jobs_by_state, job_statistics,fetch_job
from AsyncDistribJobs.operations import process_job

import argparse
import pickle


def external_inference(drugID,
                       dataset=None,
                       external_dataset=None,
                       results_path=None,
                       models_path=None,
                       return_stds=False,
                       **kwargs):

    #obtain the best parameters from the search object
    with open(Path('/home/fcarli/CellHit/results/gdsc/search_and_inference/moa_primed/models')/f'{int(drugID)}.pkl', 'rb') as f:
        best_params = pickle.load(f)['best_params']
        best_params['device'] = f"cuda:{kwargs['gpu_id']}"

    data_dict = prepare_data(drugID=drugID,
                             dataset=dataset,
                             use_external_datasets=True,
                             external_dataset=external_dataset,
                             gene_selection_mode='moa_primed',
                             **kwargs)
    
    results = inference(best_params=best_params,
                        refit=True, internal_inference=False, 
                        gene_selection_mode='moa_primed',
                        return_shaps=True, 
                        return_model=True,  
                        return_stds=return_stds,
                        fix_seed=True,
                        **data_dict)
    
    #create the dataframe of the predictions
    predictions_df = pd.DataFrame()
    predictions_df['DrugName'] = [data_dict['loader'].get_drug_name(drugID)]*len(results['predictions'])
    predictions_df['DrugID'] = [drugID]*len(results['predictions'])
    predictions_df['Index'] = data_dict['external_indexes']
    predictions_df['Predictions'] = results['predictions']
    #unstandardize the predictions
    mean = data_dict['loader'].get_drug_mean(drugID)
    std = data_dict['loader'].get_drug_std(drugID)

    if return_stds:
        predictions_df['Stds'] = results['std']
        predictions_df['Stds'] = predictions_df['Stds']*std

    predictions_df['Predictions'] = predictions_df['Predictions']*std + mean
    predictions_df['Source'] = data_dict['loader'].get_indexes_sources(data_dict['external_indexes'].tolist())
    predictions_df.to_csv(Path(results_path)/dataset/'predictions'/f'{drugID}.csv',index=False)

    #create the dataframe of the shap values
    shap_df = pd.DataFrame()
    #values are inside results['shap_values'].values, whereas columns are inside results['shap_values'].feature_names
    shap_df['Index'] = data_dict['external_indexes']
    shap_df['Source'] = data_dict['loader'].get_indexes_sources(data_dict['external_indexes'].tolist())
    shap_df = pd.concat([shap_df, pd.DataFrame(results['shap_values'].values, columns=results['shap_values'].feature_names)], axis=1)
    shap_df = pd.DataFrame(results['shap_values'].values, columns=results['shap_values'].feature_names)
    shap_df['Index'] = data_dict['external_indexes']
    shap_df = shap_df.set_index('Index')
    shap_df.to_csv(Path(results_path)/dataset/'shaps'/f'{drugID}.csv')#,index=False)

    if kwargs['return_shaps']:
        
        with open(Path(results_path) / dataset / 'raw_shaps' / f'{drugID}.pkl', 'wb') as f:
            pickle.dump(results['shap_values'], f)


def run_full_asynch_inference(args,inference_database_path):

    engine = create_engine(f'sqlite:///{inference_database_path}/models_inference_{args["dataset"]}.db')
    configure_database(engine,reset=args["build_inference_db"])

    #if specified, build the search database
    if args["build_inference_db"]:

        if args["dataset"] == 'gdsc':

            #populate the search database
            loader = DatasetLoader(dataset=args['dataset'],
                                    data_path=args['data_path'],
                                    celligner_output_path=args['celligner_output_path'],
                                    use_external_datasets=True,
                                    samp_x_tissue=2,
                                    random_state=args['random_state'])
            
            drugs_ids = loader.get_drugs_ids()
        

        #if dataset is PRISM, inference only on drugs with corr > 0.2
        elif args['dataset'] == 'prism':
            perfs = pd.read_csv(args['tabs_path']/'drugs_performances_PRISM.tsv',sep='\t')[['DrugID','MOA Corr']]
            drugs_ids = perfs[perfs['MOA Corr'] >= 0.2]['DrugID'].tolist()

            #TODO: REMEMBER TO REMOVE THIS
            #remove the ones already present in the direcoe
            processed = [i.stem for i in Path('/home/fcarli/CellHit/results/prism/search_and_inference/moa_primed/full_inference').iterdir() if i.suffix == '.csv']
            processed = set([int(i) for i in processed])

            drugs_ids = [i for i in drugs_ids if i not in processed]

        
        #processed = [int(i.stem) for i in (Path(args['results_path'])/args['dataset']/'predictions').iterdir() if i.suffix == '.csv']
        jobs_list = [Job(state='pending',payload={'drugID': int(drugID)},cid=f'{drugID}') for drugID in drugs_ids]
        #jobs_list = [Job(state='pending',payload={'drugID': int(drugID)},cid=f'{drugID}') for drugID in drugs_ids if int(drugID) not in processed]
        add_jobs(jobs_list)


    while len(get_jobs_by_state('pending')) > 0:
        process_job(external_inference,**args)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser(description='Inference for CellHit')
    argparser.add_argument('--dataset', type=str, default='gdsc')
    argparser.add_argument('--drugID', type=int, default=1909)
    argparser.add_argument('--random_state', type=int, default=0)
    argparser.add_argument('--return_stds', type=bool, default=True)
    argparser.add_argument('--return_shaps', type=bool, default=True)
    argparser.add_argument('--cv_iterations', type=int, default=15)
    argparser.add_argument('--celligner_output_path', type=str, default='/home/fcarli/WebCellHit/data/transcriptomics/celligner_CCLE_TCGA_optimized_revised.feather')
    argparser.add_argument('--data_path', type=str, default='/home/fcarli/CellHit/data') #looks at the data path of CellHit
    argparser.add_argument('--results_path', type=str, default='/home/fcarli/WebCellHit/results/CellHit/inference')
    argparser.add_argument('--tabs_path', type=str, default='./../../tables/tabs/')
    argparser.add_argument('--database_path', type=str, default='/home/fcarli/WebCellHit/database')
    argparser.add_argument('--gpu_id', type=int, default=0)

    argparser.add_argument('--inference_mode', type=str, default='asynch')
    argparser.add_argument('--build_inference_db', default=False, action='store_true')

    args = argparser.parse_args()

    args_dict = vars(args)
    args_dict['models_path'] = Path(args.results_path) / '..' / 'models' / args.dataset
    args_dict['data_path'] = Path(args.data_path)
    args_dict['tabs_path'] = Path(args.tabs_path)
    args_dict['database_path'] = Path(args.database_path)

    if args.inference_mode == 'asynch':

        if 'drugID' in args_dict:
            del args_dict['drugID']

        run_full_asynch_inference(args_dict, args_dict['database_path'])
    
    elif args.inference_mode == 'single_drug':
        external_inference(**args_dict)


