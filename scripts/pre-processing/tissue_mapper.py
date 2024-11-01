import pandas as pd
from pathlib import Path
import argparse
import requests
import time
import json
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--directory_path', default = '~/WebCellHit/')
    parser.add_argument('--output_path', default = '~/WebCellHit/data/metadata/tissueMap.json')
    args = parser.parse_args()

    #expand user in the path
    for arg in ['directory_path','output_path']:
        if '~' in args.__dict__[arg]:
            args.__dict__[arg] = args.__dict__[arg].replace('~',os.path.expanduser('~'))

    directory_path = Path(args.directory_path)
    celligner_output_path = Path(args.celligner_output_path)

    df = pd.read_feather(celligner_output_path)

    tcga_oncontree = pd.read_csv(directory_path / "data" / "metadata" / "tcga_oncotree_data.csv")
    ccle_oncontree = pd.read_csv(directory_path / "data" / "metadata" / "Model.csv")

    mainTypeMap = {}

    for code in list(set(list(tcga_oncontree['oncotree_code'].unique())+list(ccle_oncontree['OncotreeCode'].unique()))):

        # Define the base URL and endpoint
        base_url = f'https://oncotree.mskcc.org:443/api/tumorTypes/search/code/{code}'

        # Define the query parameters
        params = {
            'exactMatch': 'true',
            'levels': '2,3,4,5'
            }

        try:
            # Make the GET request with query parameters
            response = requests.get(base_url, params=params)

            # Raise an exception for HTTP error codes (4xx and 5xx)
            response.raise_for_status()

            # Parse the JSON response
            data = response.json()

            mainTypeMap[code] = data[0]['tissue']

        except requests.exceptions.HTTPError as e:
            print(f"HTTP error occurred: {e}")
        
        # Add a wait between iterations
        time.sleep(1)  # Wait for 1 second between each request
        
    #save mainTypeMap
    with open(directory_path / "data" / "metadata" / "tissueMap.json", 'w') as f:
        json.dump(mainTypeMap, f)
