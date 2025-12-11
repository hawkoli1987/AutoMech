# ===========================
# 0. Run this inside a PBS interactive job on Hopper (qsub -I ...)
# ===========================

# Paths
export SHARED_FS="/scratch_aisg/SPEC-SF-AISG"
export SHARED_FS2="/scratch/Projects/SPEC-SF-AISG"
SQSH_DIR="/scratch_aisg/SPEC-SF-AISG/sqsh"
LOG_DIR="${SQSH_DIR}/logs"
mkdir -p "${SQSH_DIR}" "${LOG_DIR}"

BASE_SQSH="${SQSH_DIR}/vllm-openai-v0.11.2.sqsh"

# === 1. Create a writable build container from this base image ===
enroot create --name vllm_mech "${BASE_SQSH}"

# ===========================
# 3) Prepare an RC script to bypass the default 'vllm serve' entrypoint
#    and exec whatever command we pass (e.g. /bin/bash)
# ===========================
RC_SCRIPT="${LOG_DIR}/enroot_rc.sh"
cat > "${RC_SCRIPT}" << 'RCEOF'
#!/bin/bash
# Simple RC script: just exec the given command instead of the image entrypoint
exec "$@"
RCEOF
chmod +x "${RC_SCRIPT}"

export ENROOT_DATA_PATH="${SHARED_FS}/cache/.enroot/data/"
mkdir -p "${ENROOT_DATA_PATH}"

# ===========================
# 4) Start the build container INTERACTIVELY with bash instead of 'vllm serve'
#    You will be dropped into /bin/bash as root inside vllm_mech_build.
# ===========================
enroot start \
  --root \
  --rw \
  --rc "${RC_SCRIPT}" \
  --mount "${HOME}:${HOME}" \
  --mount "${SHARED_FS}:${SHARED_FS}" \
  --mount "${SHARED_FS2}:${SHARED_FS2}" \
  vllm_mech \
  /bin/bash

# === NOW YOU ARE INSIDE THE CONTAINER, AS ROOT, WITH A BASH SHELL ===
# Inside the container, run (manually):

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y \
    software-properties-common \
    ca-certificates \
    wget git vim \
    freecad \
    calculix-ccx calculix-cgx \
    gmsh \
    python3-pip python3-dev \
    python3-numpy python3-scipy python3-matplotlib \
    python3-pyqt5

# Python packages (include wandb and pandas as requested)
pip3 install --no-cache-dir \
    cadquery \
    ipython \
    jupyter \
    meshio \
    numpy-stl \
    pyvista \
    wandb \
    pandas

# Optional: quick sanity checks
which freecadcmd
which ccx
which gmsh
python3 -c "import vllm; print('vLLM import OK')"

# When done, exit the container:
exit

# ===========================
# 5) After you exit the container, EXPORT the modified rootfs as a new sqsh
# ===========================
FINAL_SQSH="${SQSH_DIR}/vllm_mech_v1.sqsh"
enroot export --output "${FINAL_SQSH}" vllm_mech

# ===========================
# 6) Start the container interactively with bash
# ===========================
RC_SCRIPT="${LOG_DIR}/enroot_rc.sh"
cat > "${RC_SCRIPT}" << 'RCEOF'
#!/bin/bash
# Simple RC script: just exec the given command instead of the image entrypoint
exec "$@"
RCEOF
chmod +x "${RC_SCRIPT}"

export ENROOT_DATA_PATH="${SHARED_FS}/cache/.enroot/data/"
mkdir -p "${ENROOT_DATA_PATH}"

enroot start \
  --root \
  --rw \
  --rc "${RC_SCRIPT}" \
  --mount "${HOME}:${HOME}" \
  --mount "${SHARED_FS}:${SHARED_FS}" \
  --mount "${SHARED_FS2}:${SHARED_FS2}" \
  vllm_mech \
  /bin/bash
