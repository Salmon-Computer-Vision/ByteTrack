#!/usr/bin/env bash
input=$1
prefix=$2
fps=$3
raspi_ip=$4
encoding=$5
timezone="$6"
docker_image=bytetrack:manual

sshfs -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -o IdentityFile=~/.ssh/revtunnel_id_rsa \
    lockedsaphen@${raspi_ip}:/media/usb0/ track_outputs/

cd track_outputs

export TZ="$timezone"

../../deploy/TensorRT/cpp/build/bytetrack ../yolox_nano_salmon/model_trt.engine -i "${input}" "$prefix" $fps $encoding
