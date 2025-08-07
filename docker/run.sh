#!/usr/bin/env bash

set -exu

# CONFIGURE DATA PATHS (please modify according to your environment)
IG_PATH="/path/to/isaacgym"
CACHE_PATH="/path/to/.cache"
DATA_PATH="/path/to/data"
TMP_PATH="/path/to/tmp"

# Figure out repository root.
SCRIPT_DIR="$( cd "$( dirname $(realpath "${BASH_SOURCE[0]}") )" && pwd )"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"

# Create a temporary directory to be shared between host<->docker.
mkdir -p "${CACHE_PATH}"
# TMP_PATH used to be a fixed one at host machine's /tmp/docker/
mkdir -p "${TMP_PATH}"

# Launch docker with the following configuration:
# * Display/Gui connected
# * Network enabled (passthrough to host)
# * Privileged
# * GPU devices visible
# * Current working git repository mounted at ${HOME}
# * 8Gb Shared Memory
# NOTE: comment out `--network host` for profiling with `nsys-ui`.

# NOTE: configure container's name with `--name`  
docker run -it \
    --name dywa_1 \
    --mount type=bind,source="${REPO_ROOT}",target="/home/user/$(basename ${REPO_ROOT})" \
    --mount type=bind,source="${IG_PATH}",target="/opt/isaacgym/" \
    --mount type=bind,source="${CACHE_PATH}",target="/home/user/.cache/pkm" \
    --mount type=bind,source="${DATA_PATH}",target="/input" \
    --mount type=bind,source="${TMP_PATH}",target="/tmp/docker" \
    --shm-size=32g \
    --network host \
    --privileged \
    --gpus all \
    "$@" \
    "pkm1:v0"
