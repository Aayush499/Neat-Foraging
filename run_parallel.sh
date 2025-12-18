#!/bin/bash
#SBATCH --job-name=neat_arr
#SBATCH --ntasks=1              # 1 task = 1 Python process
#SBATCH --cpus-per-task=16      # how many cores that process can use
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --array=2-4
#SBATCH --output=logs/neat_%A_%a.out

module load Miniconda3
source /sw/eb/sw/Miniconda3/24.11.1/etc/profile.d/conda.sh
conda activate neat

SUB="parallel_test_${SLURM_ARRAY_TASK_ID}"
python3 main_parallel.py --sub "$SUB"

