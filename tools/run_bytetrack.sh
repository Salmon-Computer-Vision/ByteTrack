#!/usr/bin/env bash
input=$1
prefix=$2
fps=$3
raspi_ip=$4
docker_image=bytetrack:manual

mkdir -p track_outputs
sshfs lockedsaphen@${raspi_ip}:/media/usb0/ track_outputs/ -o IdentityFile=~/.ssh/revtunnel_id_rsa

cd track_outputs

../../deploy/TensorRT/cpp/build/bytetrack ../yolox_nano_salmon/model_trt.engine -i "${input}" "$prefix" $fps
