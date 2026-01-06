## This script stays in the home directory and is called by SLURM to run the job.
## It should get the code to the scratch, setup the environment, and run the job.
#! /bin/bash
#check if projects dir exists
if [ ! -d "/scratch/$USER/projects" ]; then
    mkdir -p /scratch/$USER/projects
fi
cd /scratch/$USER/projects
#check if ram dir exists else git pull it
if [ ! -d "/scratch/$USER/projects/ram" ]; then
    git clone git@github.com:sombit888/ram.git
else
    cd /scratch/$USER/projects/ram
    git pull
fi
scratch_dir=/scratch/$USER/projects/ram
cd $scratch_dir
# check if .venv exists else setup environment
if [ ! -d "$scratch_dir/.venv" ]; then
    bash requirements/slurm_env.sh ${1:-"openvla"} $3
    # bash requirements/install.sh openvla
fi

source .venv/bin/activate
bash examples/embodiment/run_embodiment.sh ${2:-maniskill_ppo_openvla_quickstart}
# bash examples/embodiment/run_embodiment.sh $2

# bash slurm_scripts/run_job.sh ${1:-"openvla"}