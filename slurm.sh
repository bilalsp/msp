#!/bin/sh
#SBATCH --job-name=MSP_Exact
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --nodelist=cn-003
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1 # number of tasks per node
#SBATCH --cpus-per-task=44  # number of cores per task 
#SBATCH --gpus=1
#SBATCH --output=./rand_out/training-jobID.%j.out
#SBATCH --error=./rand_out/training-jobID.%j.err
#SBATCH --time=1-20:00:00

module purge
module load CUDA/11.2 Python/Anaconda_v11.2020 gnu8/8.2.0

source deactivate
source activate /home/mansari/.conda/envs/tf2-gpu


python -u bin/main.py --config_path=bin/configs/random_solver_msp_25_5.yml --action='run_solver'
