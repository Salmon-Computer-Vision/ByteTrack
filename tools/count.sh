#!/usr/bin/env bash

python3 tools/demo_count.py video -f exps/example/mot/yolox_nano_mix_det.py --path "$1" \
    -c pretrained/bytetrack_nano_salmon_epoch7_ckpt.pth.tar --aspect_ratio_thresh 0.9 --fp16 --fuse --save_result
