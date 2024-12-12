#!/bin/bash

#SBATCH --nodes=1
#SBATCH --array=0-99
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu2080
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=00:15:00

#SBATCH --job-name=XLNet-Classification
#SBATCH --output=logs/%x-%A-partition_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=g_krus03@uni-muenster.de

module purge
module load palma/2024a
module load GCCcore/13.3.0
module load Python/3.12.3

INPUT_FILE="data/20240823_Applications_Grants_Combined/partition_${SLURM_ARRAY_TASK_ID}.csv"

srun venv/bin/python train_transf_class_newdata.py --input-file "$INPUT_FILE" --partition-number ${SLURM_ARRAY_TASK_ID}
