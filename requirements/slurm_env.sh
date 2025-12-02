# curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/scratch/$USER/uv sh

# make it conditional if uv not installed
# mkdir -p /scratch/$USER/projects/
# cd /scratch/$USER/projects/RLinf 
# cd RLinf && git pull
if ! command -v uv &> /dev/null
then
    curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/scratch/$USER/uv sh
    source /scratch/$USER/uv/env
fi
# Python bytecode cache
export PYTHONPYCACHEPREFIX=/scratch/$USER/.pycache

# UV cache and local Python installations
export UV_HOME=/scratch/$USER/.local/share/uv
export UV_VENV_DIR=/scratch/$USER/.venv
export UV_HOME=/scratch/$USER/.local/share/uv

# Generic Python temp and cache dirs
export XDG_DATA_HOME=/scratch/$USER/.local/share  # most important
export XDG_CACHE_HOME=/scratch/$USER/.cache
export PIP_CACHE_DIR=/scratch/$USER/.cache/pip
export TORCH_HOME=/scratch/$USER/.cache/torch
export HF_HOME=/scratch/$USER/.cache/huggingface
export OMNIGIBSON_DATASET_PATH=/scratch/$USER/datasets/behavior-1k-assets

# Optional: temp dir for any build
export TMPDIR=/scratch/$USER/tmp
mkdir -p $TMPDIR
bash requirements/install.sh ${1:-"openvla"}
hf download gen-robot/openvla-7b-rlvla-warmup --local-dir /scratch/sombit_dey/openvla-7b-rlvla-warmup/

