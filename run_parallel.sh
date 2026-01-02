#!/bin/bash
#SBATCH --job-name=neat_arr
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --array=1-6
#SBATCH --output=logs/neat_%A_%a.out

module load Miniconda3
source /sw/eb/sw/Miniconda3/24.11.1/etc/profile.d/conda.sh
conda activate neat

case "$SLURM_ARRAY_TASK_ID" in
  1)
    python3 main_parallel.py --sub basic_test_1
    ;;
  2)
    python3 main_parallel.py --sub basic_test_2
    ;;
  3)
    python3 main_parallel.py --sub basic_test_3
    ;;
  4)
    python3 main_parallel.py --sub basic_test_1 --network recursive
    ;;
  5)
    python3 main_parallel.py --sub basic_test_2 --network recursive
    ;;
  6)
    python3 main_parallel.py --sub basic_test_3 --network recursive
    ;;
esac
