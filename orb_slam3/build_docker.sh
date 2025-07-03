#!/bin/bash

IMAGE_NAME=orb_slam3_cpu

echo "[+] Removing old Docker image if exists..."
docker rmi -f $IMAGE_NAME

echo "[+] Building Docker image: $IMAGE_NAME"
docker build -t $IMAGE_NAME .

echo "[+] Done. You can now run it with:"
echo "    docker run --rm -it --privileged \\"
echo "        --env=\"DISPLAY\" \\"
echo "        --volume=\"/tmp/.X11-unix:/tmp/.X11-unix:rw\" \\"
echo "        $IMAGE_NAME /bin/bash"

