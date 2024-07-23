#!/bin/bash -l
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=40g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alokshah@seas.upenn.edu
#SBATCH -p gpu
#SBATCH --gres=gpu:a40:1

cd /home/gg8/vlm/WebSAM-Adapter

# Initialize conda
source /home/gg8/miniconda3/etc/profile.d/conda.sh

# Activate the conda environment
conda activate mlrclub

# Check GPU allocation
echo "Checking GPU allocation..."
nvidia-smi


# Run your training script
python test.py
