import numpy as np
import pandas as pd

def elaborate_output(predictions,model,data,
                     drug_id,drug_name,
                     mean_mapper,std_mapper,
                     repurposing_target,
                     source=None,
                     topk=15):

    gene_names = np.array(model.models[0].feature_names)

    if 'index' in data.columns:
        indexes = data['index'].values
    else:
        indexes = data.index.values

    preds = predictions['predictions']
    stds = predictions['std']

    #get shap values
    shaps = predictions['shap_values'].values

    #prediction dataframe
    preds_df = pd.DataFrame()
    
    if source is not None:
        preds_df['Source'] = source

    preds_df['DrugID'] = [drug_id]*len(indexes)
    preds_df['DrugName'] = [drug_name]*len(indexes)
    preds_df['index'] = indexes
    preds_df['prediction'] = (preds * std_mapper[drug_name]) + mean_mapper[drug_name]
    preds_df['std'] = stds * std_mapper[drug_name]

    #take topk 15 absolute values for each instance (most important features)
    topk_indexes = np.abs(shaps).argsort(axis=1)[:,-topk:]
    #transform the indexes into feature names
    topk_feature_names = gene_names[topk_indexes]
    #shap values
    values = shaps[np.arange(shaps.shape[0])[:,None], topk_indexes]
    #transform the bidimensional array into a list of lists
    topk_feature_names = topk_feature_names.tolist()

    #create a new column with the topk feature names
    preds_df['TopGenes'] = topk_feature_names
    preds_df['TopGenes'] = preds_df['TopGenes'].apply(lambda x: ','.join(x))
    preds_df['ShapDictionary'] = [dict(zip(topk_feature_names[i], values[i])) for i in range(values.shape[0])]
    preds_df['PutativeTarget'] = [repurposing_target]*len(indexes)

    return preds_df