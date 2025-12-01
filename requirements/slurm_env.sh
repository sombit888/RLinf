curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/scratch/$USER/uv sh

export UV_CACHE_DIR="/scratch/$USER/.cache/uv"

# pip cache
export PIP_CACHE_DIR="/scratch/$USER/.cache/pip"

# Python temp builds
export TMPDIR="/scratch/$USER/tmp"
export TEMP="/scratch/$USER/tmp"
export TMP="/scratch/$USER/tmp"

# XDG cache
export XDG_CACHE_HOME="/scratch/$USER/.cache"
