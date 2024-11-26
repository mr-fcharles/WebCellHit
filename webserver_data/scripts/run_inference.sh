#!/bin/bash
#PBS -l select=1:ncpus=52:mem=180gb
#PBS -q q07anacreon
#PBS -N gdsc_final_inference

cd /home/fcarli/WebCellHit/webserver_data/scripts
source activate /home/fcarli/.conda/envs/provenv

python run_inference_memory.py --dataset gdsc
