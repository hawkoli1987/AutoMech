#!/bin/bash
set -e

export SHARED_FS="/scratch_aisg/SPEC-SF-AISG"
export SHARED_FS2="/scratch/Projects/SPEC-SF-AISG"
SQSH_DIR="/scratch_aisg/SPEC-SF-AISG/sqsh"
SQSH_FILE="${SQSH_DIR}/vllm_mech_v1.sqsh"

# Set enroot directories to avoid permission issues
export ENROOT_DATA_PATH="${SHARED_FS}/.enroot/data"
export ENROOT_RUNTIME_PATH="${SHARED_FS}/.enroot/runtime"
mkdir -p "${ENROOT_DATA_PATH}" "${ENROOT_RUNTIME_PATH}"

export CONTAINER_NAME="vllm_mech"
# Create container if it doesn't exist
if ! enroot list | grep -q "${CONTAINER_NAME}"; then
    echo "Creating enroot container: ${CONTAINER_NAME}"
    enroot create -n "${CONTAINER_NAME}" "${SQSH_FILE}"
fi

# Server connection
export SERVER_NODE="${SERVER_NODE:-hopper-45}"
export OPENAI_API_BASE="http://${SERVER_NODE}:8001"
export OPENAI_API_BASE2="http://${SERVER_NODE}:8002"

# Build enroot --env arguments
env_args=""
prefixes=("NCCL" "CUDA" "SHARED" "HF" "WANDB" "XDG" "LOG" "CACHE" "TORCH" "TRITON" "VLLM" "OPENAI" "SERVER")

for prefix in "${prefixes[@]}"; do
    while IFS= read -r var; do
        if [ -n "${!var}" ]; then
            env_args="${env_args} --env=${var}=${!var}"
        fi
    done < <(env | grep "^${prefix}_" | cut -d'=' -f1)
done

# Create rc script to bypass container entrypoint
RC_SCRIPT="/tmp/enroot_rc_$$.sh"
cat > "${RC_SCRIPT}" << 'RCEOF'
#!/bin/bash
exec "$@"
RCEOF
chmod +x "${RC_SCRIPT}"

# Create startup script in HOME (accessible inside container via mount)
STARTUP_SCRIPT="${HOME}/.container_startup_$$.sh"
cat > "${STARTUP_SCRIPT}" << 'STARTUPEOF'
#!/bin/bash
echo "=================================================="
echo "Checking vLLM servers..."
echo "=================================================="

# Check LLM server
LLM_MODEL=$(curl -s ${OPENAI_API_BASE}/v1/models 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'] if d.get('data') else '')" 2>/dev/null)
if [ -n "$LLM_MODEL" ]; then
    echo "✓ LLM Server (${OPENAI_API_BASE}): ${LLM_MODEL}"
else
    echo "✗ LLM Server (${OPENAI_API_BASE}): Not available"
fi

# Check VLM server
VLM_MODEL=$(curl -s ${OPENAI_API_BASE2}/v1/models 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'] if d.get('data') else '')" 2>/dev/null)
if [ -n "$VLM_MODEL" ]; then
    echo "✓ VLM Server (${OPENAI_API_BASE2}): ${VLM_MODEL}"
else
    echo "✗ VLM Server (${OPENAI_API_BASE2}): Not available"
fi

echo "=================================================="
echo ""

# Start interactive bash
exec /bin/bash
STARTUPEOF
chmod +x "${STARTUP_SCRIPT}"

echo "=================================================="
echo "Starting enroot container: ${CONTAINER_NAME}"
echo "Server: ${SERVER_NODE}:8001 and ${SERVER_NODE}:8002"
echo "=================================================="

# Start container with startup script
exec enroot start \
    --root \
    --rw \
    --rc "${RC_SCRIPT}" \
    --mount "${HOME}:${HOME}" \
    --mount "${SHARED_FS}:${SHARED_FS}" \
    --mount "${SHARED_FS2}:${SHARED_FS2}" \
    ${env_args} \
    ${CONTAINER_NAME} \
    /bin/bash "${STARTUP_SCRIPT}"
