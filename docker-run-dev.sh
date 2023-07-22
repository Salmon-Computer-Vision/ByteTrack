#!/usr/bin/env bash
name=$1

# Variables
prefix=river
fps=10
raspi_ip=192.168.1.98
encoding=-5 # Set -5 if using h265
timezone="GMT+8"

# Constants
docker_image=masami.hopto.org:23411/bytetrack:dev
bytetrack_home=/home/salmonjetson/ByteTrack
workspace=/ByteTrack/YOLOX_outputs
workspace_dev=/mot_dev

mkdir -p "${bytetrack_home}"/YOLOX_outputs/track_outputs
sudo chattr +i "${bytetrack_home}"/YOLOX_outputs/track_outputs

# Run ByteTrack on input outputting to the YOLOX_outputs folder into the $prefix folder
sudo docker run -it --rm --runtime nvidia \
    -v ${bytetrack_home}/pretrained:/ByteTrack/pretrained \
    -v ${bytetrack_home}/datasets:/ByteTrack/datasets \
    -v ${bytetrack_home}/YOLOX_outputs:/ByteTrack/YOLOX_outputs \
    -v /tmp/.X11-unix/:/tmp/.X11-unix:rw \
    -v /home/salmonjetson/.ssh:/home/user/.ssh \
    -v ${bytetrack_home}:${workspace_dev} \
    -w "$workspace_dev" \
    --device /dev/video0:/dev/video0:mwr \
    --net=host \
    --name "$name" \
    -e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
    -e DISPLAY=$DISPLAY \
    --privileged \
    $docker_image
