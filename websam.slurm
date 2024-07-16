#!/bin/bash -l
#SBATCH --time=08:00:00

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=4

#SBATCH --mem=32g

#SBATCH --mail-type=ALL

#SBATCH --mail-user=alokshah@seas.upenn.edu

#SBATCH -p gpu

#SBATCH --gres=gpu:a5000:1

cd /home/alokshah/apps/WebSAM-Adapter

export PATH=/home/alokshah/miniconda3/envs/websam_env/bin:$PATH

python train.py