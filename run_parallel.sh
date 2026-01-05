#!/bin/bash
#SBATCH --job-name=neat_arr
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=05:00:00
#SBATCH --mem=16G
#SBATCH --array=1-9
#SBATCH --output=logs/neat_%A_%a.out

module load Miniconda3
source /sw/eb/sw/Miniconda3/24.11.1/etc/profile.d/conda.sh
conda activate neat

case "$SLURM_ARRAY_TASK_ID" in
  1)
    python3 main_parallel.py --sub ff_data_collection_1 --pheromone_receptor true --network ff
    ;;
  2)
    python3 main_parallel.py --sub ff_data_collection_2 --pheromone_receptor true --network ff
    ;;
  3)
    python3 main_parallel.py --sub ff_data_collection_3 --pheromone_receptor true --network ff
    ;;
  4)
    python3 main_parallel.py --sub recursive_data_collection_1 --pheromone_receptor false --network recursive 
    ;;
  5)
    python3 main_parallel.py --sub recursive_data_collection_2 --pheromone_receptor false --network recursive
    ;;
  6)
    python3 main_parallel.py --sub recursive_data_collection_3 --pheromone_receptor false --network recursive
    ;;
  7)
    python3 main_parallel.py --sub recursive_pheromone_data_collection_1 --pheromone_receptor true --network recursive
    ;;
  8)
    python3 main_parallel.py --sub recursive_pheromone_data_collection_2 --pheromone_receptor true --network recursive
    ;;
  9)
    python3 main_parallel.py --sub recursive_pheromone_data_collection_3 --pheromone_receptor true --network recursive
    ;;
  *)
    echo "Invalid SLURM_ARRAY_TASK_ID"
    ;;
esac
