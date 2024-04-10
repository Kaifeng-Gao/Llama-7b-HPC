#!/bin/bash

#SBATCH --job-name=spider
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --mail-type=ALL
#SBATCH --gpus=a100:1

module purge
module load miniconda
conda activate Llama

python finetune_spider.py
