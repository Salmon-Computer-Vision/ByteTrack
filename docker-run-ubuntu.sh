#!/usr/bin/env bash

name=$1

# Constants
docker_image=bytetrack_ubuntu
bytetrack_home=~/ByteTrack
workspace=/workspace/mot_dev

docker run -it --rm --runtime nvidia \
    -v "${bytetrack_home}:${workspace}" \
    -w "$workspace" \
    --net host --name bytetrack --privileged \
    -e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
    $docker_image
