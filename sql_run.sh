#!/bin/bash

#SBATCH --job-name=sql
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --mail-type=ALL

module purge
module load miniconda
conda activate llama

python fine_tune.py