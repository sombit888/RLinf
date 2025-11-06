# Set a persistent base dir for all ManiSkill and PhysX assets
export SAPIEN_HOME=/scratch/$USER/sapien
export MANISKILL_ASSET_DIR=/scratch/$USER/maniskill_assets
export PHYSX_VERSION=105.1-physx-5.3.1.patch0
export PHYSX_DIR=$SAPIEN_HOME/physx/$PHYSX_VERSION

# Make sure directories exist
mkdir -p "$PHYSX_DIR" "$MANISKILL_ASSET_DIR"

# Download ManiSkill assets to the custom directory
python -m mani_skill.utils.download_asset bridge_v2_real2sim -y --dir "$MANISKILL_ASSET_DIR"
python -m mani_skill.utils.download_asset widowx250s -y --dir "$MANISKILL_ASSET_DIR"

# Download and unpack PhysX binaries into the scratch space
wget -O "$PHYSX_DIR/linux-so.zip" \
  "https://github.com/sapien-sim/physx-precompiled/releases/download/$PHYSX_VERSION/linux-so.zip"

unzip "$PHYSX_DIR/linux-so.zip" -d "$PHYSX_DIR"
rm "$PHYSX_DIR/linux-so.zip"
