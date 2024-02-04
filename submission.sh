#!/bin/bash
#SBATCH --gres=gpu:teslaa40:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mc620
export PATH=/vol/bitbucket/mc620/DeepLearningCW1/venv/bin/:$PATH

source /vol/bitbucket/mc620/DeepLearningCW1/venv/bin/activate
source activate
source /vol/cuda/12.0.0/setup.sh
/usr/bin/nvidia-smi
uptime

python -m src.main
