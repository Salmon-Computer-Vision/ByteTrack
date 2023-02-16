#!/usr/bin/env bash

python3 tools/demo_track.py video -f exps/example/mot/yolox_x_mix_det.py -c pretrained/bytetrack_x_mot17.pth.tar --aspect_ratio_thresh 0.9 --fp16 --fuse --save_result
