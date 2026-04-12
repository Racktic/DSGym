#!/bin/bash
# Install dspredict containers on a new node (follows executors/README.md)
# Usage: ssh <node> && cd /data/fnie/qixin/DSGym && bash scripts/install_dspredict_containers.sh
#
# For dspredict-easy/swap/hard, only executor-kaggle is needed.
# Pass --mle to also build executor-mle (for dspredict-mledojo / mle-bench).
#
# UID/GID are matched to the host user via --build-arg so that containers
# can read/write the NFS-mounted /data directory without permission errors.

set -e

DSGYM=/data/fnie/qixin/DSGym
cd $DSGYM/executors

HOST_UID=$(id -u)
HOST_GID=$(id -g)
echo "Host user: $(id -un) UID=$HOST_UID GID=$HOST_GID"

BUILD_MLE=false
if [ "$1" = "--mle" ]; then
    BUILD_MLE=true
fi

echo "=============================="
echo "Step 1: Build manager-prebuilt"
echo "=============================="
sudo docker build -t manager-prebuilt ./manager/

echo "=============================="
echo "Step 2: Build executor-kaggle (for dspredict-easy/swap/hard)"
echo "=============================="
sudo docker build \
    --build-arg HOST_UID=$HOST_UID \
    --build-arg HOST_GID=$HOST_GID \
    -t executor-kaggle \
    ./container_images/kaggle_image/

if [ "$BUILD_MLE" = true ]; then
    echo "=============================="
    echo "Step 3: Build executor-mle (for dspredict-mledojo)"
    echo "=============================="
    cd $DSGYM
    sudo docker build \
        --build-arg HOST_UID=$HOST_UID \
        --build-arg HOST_GID=$HOST_GID \
        -t executor-mle \
        -f executors/container_images/mle_image/Dockerfile .
    cd $DSGYM/executors
fi

echo "=============================="
echo "Done. Built images:"
echo "=============================="
sudo docker images | grep -E "executor-kaggle|executor-mle|manager-prebuilt"

echo ""
echo "Next: start containers"
echo "  cd $DSGYM/executors"
echo "  sudo docker compose -f docker-dspredict-easy.yml up -d    # Easy/Swap"
echo "  sudo docker compose -f docker-dspredict-hard.yml up -d    # Hard"
if [ "$BUILD_MLE" = true ]; then
    echo "  sudo docker compose -f docker-dspredict-mledojo.yml up -d # MLE Dojo"
fi
