#!/bin/sh
#BATCH --job-name="first-nnUNet-gpu"
#SBATCH --partition=ird_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --time=5:59:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-user=alexandre.devauxriviere@gmail.com
#SBATCH --mail-type=ALL

module () {
    eval `/usr/bin/modulecmd bash $*`
}

module load bioinfo-cirad
module load pytorch/2.x

export PATH=$PATH:/usr/local/cuda/bin
export INCLUDE=$INCLUDE:/usr/local/cuda/include
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda/lib64

export nnUNet_data_base="/home/user-data_challenge-22/scratch/PINKCC_challenge_2025"
export nnUNet_raw="${nnUNet_data_base}/nnUNet_raw_data"
export nnUNet_preprocessed="${nnUNet_data_base}/nnUNet_preprocessed"
export nnUNet_results="${nnUNet_data_base}/nnUNet_results"

source /home/user-data_challenge-22/scratch/miniconda3/etc/profile.d/conda.sh
conda activate /home/user-data_challenge-22/scratch/miniconda3/envs/.venv/

/home/user-data_challenge-22/scratch/miniconda3/envs/.venv/bin/python /lustre/user-data_challenge-22/PINKCC_challenge_2025/ocd/dataset/nnUNet/convert_dataset.py

