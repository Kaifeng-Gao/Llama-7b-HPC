#!/bin/bash

#SBATCH --job-name=sql
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --mail-type=ALL
#SBATCH --gpus=a100:1

module purge
module load miniconda
conda activate llama

python finetune.py
