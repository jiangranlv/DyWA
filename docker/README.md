# Docker Configuration

## Instructions

### Install Docker

Install docker in your host system by following the instructions [here](https://docs.docker.com/engine/install/ubuntu/ ).
Afterward, follow the post-install instructions [here](https://docs.docker.com/engine/install/linux-postinstall/).
This step is required to avoid using the `sudo` user during docker builds and runs, as we transfer the permissions
of the host user to the docker user for display.

Depending on your system, you may also need [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and [Docker Buildkit](https://docs.docker.com/build/buildkit/).


### Build Docker Image

Inspect the contents of [build.sh](./build.sh) and the [Dockerfile](Dockerfile), then run:

```bash
./build.sh
```

This process may take around 20 minuets, and consume about 13.7 GB of disk-space to store the docker image.

Note that we assume availability of NVIDIA GPUs during docker build _and_ deployment.

### Run Docker Image

Before running the docker image, check that the following directories are available in your host system:

```
IG_PATH="/home/DyWA/isaacgym"
CACHE_PATH="/home/DyWA/.cache/dywa"
DATA_PATH="/home/DyWA/datasets/"
```

Otherwise, configure the directories in [run.sh](./run.sh) so that `${IG_PATH}` refers to a directory
that exists in your filesystem. We use these directories to share persistent assets with the
docker containers to avoid oversizing the images and reusing these assets across different runs.

Afterward, inspect the contents of [run.sh](./run.sh). If it looks good, start the docker container:

```bash
./run.sh
```

By default, this will start you with a `bash` shell inside the container.
Afterward, return to the main instructions and follow the instructions there!
