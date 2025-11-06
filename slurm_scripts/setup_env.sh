#! /bin/bash
scratch_dir=/scratch/$USER/projects
mkdir -p $scratch_dir
cd $scratch_dir
git clone git@github.com:sombit888/RLinf.git
cd RLinf 
bash requirements/install.sh openvla
source .venv/bin/activate 