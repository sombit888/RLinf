#!/bin/bash

# Embodied dependencies for cluster environments
# This script checks for required system packages without attempting to install them

echo "Checking system dependencies for embodied environments..."

# List of required packages - these should be pre-installed on the cluster
REQUIRED_PACKAGES=(
    "wget"
    "unzip" 
    "curl"
    "libavutil-dev"
    "libavcodec-dev"
    "libavformat-dev"
    "libavdevice-dev"
    "libibverbs-dev"
    "mesa-utils"
    "libosmesa6-dev"
    "freeglut3-dev"
    "libglew-dev"
    "libegl1"
    "libgles2"
    "libglvnd-dev"
    "libglfw3-dev"
    "libgl1-mesa-dev"
    "libgl1-mesa-glx"
    "libglib2.0-0"
    "libsm6"
    "libxext6"
    "libxrender-dev"
    "libgomp1"
)

missing_packages=()

# Check for required system libraries
for pkg in "${REQUIRED_PACKAGES[@]}"; do
    case "$pkg" in
        *-dev)
            # For development packages, check if header files exist
            pkg_name=$(echo "$pkg" | sed 's/-dev$//')
            if ! pkg-config --exists "$pkg_name" 2>/dev/null && ! ldconfig -p | grep -q "$pkg_name" 2>/dev/null; then
                missing_packages+=("$pkg")
            fi
            ;;
        *)
            # For regular packages, check if they're available
            if ! command -v "$pkg" >/dev/null 2>&1 && ! ldconfig -p | grep -q "$pkg" 2>/dev/null; then
                missing_packages+=("$pkg")
            fi
            ;;
    esac
done

if [ ${#missing_packages[@]} -gt 0 ]; then
    echo "Warning: The following system packages may be missing on this cluster:"
    printf "  %s\n" "${missing_packages[@]}"
    echo ""
    echo "Please contact your cluster administrator to install these packages:"
    echo "  sudo apt-get install -y ${missing_packages[*]}"
    echo ""
    echo "Continuing installation - some features may not work without these packages."
else
    echo "All required system dependencies appear to be available."
fi

# Check for graphics/rendering capabilities
echo "Checking graphics capabilities..."

# Check for NVIDIA driver
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "✓ NVIDIA driver detected"
    nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1
else
    echo "⚠ NVIDIA driver not detected - GPU acceleration may not be available"
fi

# Check for Vulkan
if [ -d "/etc/vulkan/icd.d" ]; then
    echo "✓ Vulkan ICD directory found"
else
    echo "⚠ Vulkan ICD directory not found at /etc/vulkan/icd.d"
    echo "  Graphics rendering may be limited"
fi

# Check for OpenGL
if command -v glxinfo >/dev/null 2>&1; then
    echo "✓ OpenGL tools available"
else
    echo "⚠ OpenGL tools (glxinfo) not available"
    echo "  Install mesa-utils if needed"
fi

echo "System dependency check completed."
