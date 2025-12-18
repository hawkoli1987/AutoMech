#!/bin/bash
# Single script to launch PBS job and enter enroot container
#
# Usage: bash mech_client.sh [client_node] [server_node]
#   client_node: node number to run client on (default: 46)
#   server_node: node number where vLLM server runs (default: 45)

CLIENT_NODE="${1:-46}"
SERVER_NODE="${2:-45}"

SETUP_SCRIPT="local_model_server/mech_client_setup.sh"

echo "Launching PBS job on hopper-${CLIENT_NODE}, server on hopper-${SERVER_NODE}"

qsub -I \
    -l select=1:mem=100gb:ngpus=1:ncpus=20:host=hopper-${CLIENT_NODE} \
    -l walltime=24:00:00 \
    -q AISG_debug \
    -N mech_client \
    -v SERVER_NODE=hopper-${SERVER_NODE} \
    -- /bin/bash "${SETUP_SCRIPT}"
