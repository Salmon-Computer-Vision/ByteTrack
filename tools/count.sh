#!/usr/bin/env bash

# python3 tools/trt.py -f exps/example/mot/yolox_nano_salmon.py -c pretrained/bytetrack_nano_salmon_epoch4_ckpt.pth.tar

python3 tools/demo_count.py video -f exps/example/mot/yolox_nano_salmon.py --path "$1" \
     --aspect_ratio_thresh 0.9 --match_thresh 0.95 --nms 0.5 --trt --save_result
