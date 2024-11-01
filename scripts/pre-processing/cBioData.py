import pandas as pd
import numpy as np
from cbio_py import cbio_mod as cb
from tqdm.auto import tqdm
import argparse
from pathlib import Path

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--directory_path', type = str, default = '~/WebCellHit/')
    args = parser.parse_args()

    #expand user in the path
    args.directory_path = Path(args.directory_path).expanduser() if '~' in args.directory_path else Path(args.directory_path)

    directory_path = Path(args.directory_path)
    output_path = directory_path / "data" / "metadata" / "tcga_oncotree_data.csv"

    ## ----------------------------------------------
    ## Get all studies and keep only TCGA PanCancer Atlas studies
    studies = cb.getAllStudies(return_type = 'dict')
    ids = []

    for study in studies:
        study_name = study['name']
        if 'TCGA, PanCancer Atlas' in study_name:
            ids.append(study['studyId'])
    ## ----------------------------------------------

    ## ----------------------------------------------
    ## Get all patients ids for each study
    patients_ids = []
    patients_ids_dict = {}

    for id in ids:

        patients_ids_dict[id] = []

        patients = cb.getAllPatientsInStudy(id, return_type = 'dict')

        for patient in patients:
            patients_ids_dict[id].append(patient['patientId'])

    for key in patients_ids_dict:
        patients_ids_dict[key] = list(set(patients_ids_dict[key]))
    ## ----------------------------------------------

    ## ----------------------------------------------
    ## Get all samples ids for each study
    samples_ids_dictionary = {}
    sample_to_patient_map = {}

    for id in patients_ids_dict.keys():

        samples_ids_dictionary[id] = []

        samples = cb.getAllSamplesInStudy(id, return_type = 'dict')

        for sample in samples:
            samples_ids_dictionary[id].append(sample['sampleId'])
            sample_to_patient_map[sample['sampleId']] = sample['patientId']

    for key in samples_ids_dictionary:
        samples_ids_dictionary[key] = list(set(samples_ids_dictionary[key]))
    ## ----------------------------------------------

    ## ----------------------------------------------
    ## Get the mapping of patient id to cancer acronym (TCGA)
    id_to_acronym = {}

    for keys in patients_ids_dict.keys():

        for patient_id in tqdm(patients_ids_dict[keys]):

            try:
                patient = cb.getAllClinicalDataOfPatientInStudy(keys, patient_id, return_type = 'dict')

                for attribute in patient:

                    if attribute['clinicalAttributeId'] == 'CANCER_TYPE_ACRONYM':
                            id_to_acronym[patient_id] = attribute['value']
            except:
                pass
    ## ----------------------------------------------
            
    ## ----------------------------------------------
    ## Get the oncotree codes and sample types for each sample
    oncotree_codes = []
    sample_types = []
    sample_ids = []

    for id in samples_ids_dictionary.keys():

        for sample_id in tqdm(samples_ids_dictionary[id]):

            try:
                sample_data = cb.getAllClinicalDataOfSampleInStudy(id, sample_id, return_type = 'dict')

                for attribute in sample_data:
                    if attribute['clinicalAttributeId'] == 'ONCOTREE_CODE':
                        oncotree_codes.append(attribute['value'])
                        sample_ids.append(sample_id)
                    if attribute['clinicalAttributeId'] == 'SAMPLE_TYPE':
                        sample_types.append(attribute['value'])

                    

            except:
                pass
    ## ----------------------------------------------
    
    ## ----------------------------------------------
    ## Create the final dataframe
    final_df = pd.DataFrame()
    final_df['sample_id'] = sample_ids
    final_df['oncotree_code'] = oncotree_codes
    final_df['sample_type'] = sample_types
    final_df['patient_id'] = final_df['sample_id'].map(sample_to_patient_map)
    final_df['tcga_cancer_acronym'] = final_df['patient_id'].map(id_to_acronym)
    final_df = final_df[['patient_id','sample_id','sample_type','oncotree_code','tcga_cancer_acronym']]
    final_df.to_csv(output_path, index = False)
    

