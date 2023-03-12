#!/usr/bin/env bash
name=$1
input=$2
docker_image=bytetrack:manual
prefix=river
fps=15
bytetrack_home=/home/salmonjetson/ByteTrack
workspace=/ByteTrack/YOLOX_outputs

# Run ByteTrack on input outputting to the YOLOX_outputs folder into the $prefix folder
sudo docker run -i --rm --runtime nvidia \
    -v ${bytetrack_home}/pretrained:/ByteTrack/pretrained \
    -v ${bytetrack_home}/datasets:/ByteTrack/datasets \
    -v ${bytetrack_home}/YOLOX_outputs:/ByteTrack/YOLOX_outputs \
    -v /tmp/.X11-unix/:/tmp/.X11-unix:rw \
    -v /home/salmonjetson/.ssh:/home/user/.ssh \
    -w "$workspace" \
    --device /dev/video0:/dev/video0:mwr \
    --net=host \
    --name "$name" \
    -e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
    -e DISPLAY=$DISPLAY \
    --privileged \
    $docker_image \
    ../tools/run_bytetrack.sh "${input}" "$prefix" $fps
