#!/usr/bin/env bash

set -ex

IMAGE_TAG='pkm1'
VERSION_TAG='v0'

# NOTE: Set context directory relative to this file.
CONTEXT_DIR="$( cd "$( dirname $(realpath "${BASH_SOURCE[0]}") )" && pwd )"

## Build docker image.
DOCKER_BUILDKIT=1 docker build --progress=plain \
    --network host \
    --build-arg UID="$(id -u)" \
    --build-arg GID="$(id -g)" \
    -t "${IMAGE_TAG}:${VERSION_TAG}" -f ${CONTEXT_DIR}/Dockerfile ${CONTEXT_DIR}
