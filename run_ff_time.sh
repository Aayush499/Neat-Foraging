#!/bin/bash
#SBATCH --job-name=neat_main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00
#SBATCH --mem=16G
#SBATCH --array=1-9
#SBATCH --output=logs/neat_%A_%a.out

module load Miniconda3
source /sw/eb/sw/Miniconda3/24.11.1/etc/profile.d/conda.sh
conda activate neat

COMMON_ARGS="--particles 3 --movement_type holonomic \
             --generations 500 --num_runs 30 \
             --time_constant 200 \
             --sparse_reward true --extra_sparse false \
             --decay_factor 0.99 --sensor_length 40"

case "$SLURM_ARRAY_TASK_ID" in
  1)
    python3 main_parallel.py $COMMON_ARGS \
      --sub rec_no_drop_1 \
      --network recursive \
      --pheromone_receptor false \
      --storage_dir recursive_no_drop
    ;;
  2)
    python3 main_parallel.py $COMMON_ARGS \
      --sub rec_drop_1 \
      --network recursive \
      --pheromone_receptor true \
      --storage_dir recursive_dropper
    ;;
  3)
    python3 main_parallel.py $COMMON_ARGS \
      --sub ff_drop_1 \
      --network ff \
      --pheromone_receptor true \
      --storage_dir ff_dropper
    ;;
  4)
    python3 main_parallel.py $COMMON_ARGS \
      --sub rec_no_drop_2 \
      --network recursive \
      --pheromone_receptor false \
      --storage_dir recursive_no_drop
    ;;
  5)
    python3 main_parallel.py $COMMON_ARGS \
      --sub rec_drop_2 \
      --network recursive \
      --pheromone_receptor true \
      --storage_dir recursive_dropper
    ;;
  6)
    python3 main_parallel.py $COMMON_ARGS \
      --sub ff_drop_2 \
      --network ff \
      --pheromone_receptor true \
      --storage_dir ff_dropper
    ;;
  7)
    python3 main_parallel.py $COMMON_ARGS \
      --sub rec_no_drop_3 \
      --network recursive \
      --pheromone_receptor false \
      --storage_dir recursive_no_drop
    ;;
  8)
    python3 main_parallel.py $COMMON_ARGS \
      --sub rec_drop_3 \
      --network recursive \
      --pheromone_receptor true \
      --storage_dir recursive_dropper
    ;;
  9)
    python3 main_parallel.py $COMMON_ARGS \
      --sub ff_drop_3 \
      --network ff \
      --pheromone_receptor true \
      --storage_dir ff_dropper
    ;;
  *)
    echo "Invalid SLURM_ARRAY_TASK_ID"
    ;;
esac
