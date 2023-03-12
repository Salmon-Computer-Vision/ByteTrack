#!/usr/bin/env bash
input=$1
prefix=$2
fps=$3
docker_image=bytetrack:manual
raspi_ip=192.168.1.98

sshfs lockedsaphen@${raspi_ip}:/media/usb0/ track_outputs/ -o IdentityFile=~/.ssh/revtunnel_id_rsa

cd track_outputs

../../deploy/TensorRT/cpp/build/bytetrack ../yolox_nano_salmon/model_trt.engine -i "${input}" "$prefix" $fps
