#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<MY COLLEGE USERNAME>
export PATH=/vol/bitbucket/${USER}/myvenv/bin/:$PATH

source /vol/bitbucket/${USER}/myenv/bin/activate
source activate
source /vol/cuda/12.0.0/setup.sh
/usr/bin/nvidia-smi
uptime

python -m src.main
