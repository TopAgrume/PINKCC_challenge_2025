#!/bin/sh
#BATCH --job-name="test-slurm-gpu"
#SBATCH --partition=ird_gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=5:59:59
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G

module () {
    eval `/usr/bin/modulecmd bash $*`
}

module load bioinfo-cirad
module load pytorch/2.x

export PATH=$PATH:/usr/local/cuda/bin
export INCLUDE=$INCLUDE:/usr/local/cuda/include
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda/lib64


source /home/user-data_challenge-22/scratch/miniconda3/etc/profile.d/conda.sh
conda activate /home/user-data_challenge-22/scratch/miniconda3/envs/.venv/

/home/user-data_challenge-22/scratch/miniconda3/envs/.venv/bin/python /lustre/user-data_challenge-22/PINKCC_challenge_2025/scripts/train_2d_nn_unet.py
