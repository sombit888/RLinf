#!/bin/bash
#SBATCH --job-name=Maniskill_PPO_baseline    # Job name
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --cpus-per-gpu=20
#SBATCH --gres=gpu:h200:2     # Request GPUs (default to 1 if not provided)
#SBATCH --time=12:00:00        
#SBATCH --mem-per-gpu=64G
#SBATCH --output=/scratch/sombit_dey/job_%j.out
#SBATCH --error=/scratch/sombit_dey/job_%j.err

bash slurm_scripts/run_job.sh openvla maniskill_ppo_openvla_quickstart gen-robot/openvla-7b-rlvla-warmup