#!/bin/bash

# search space: basic block search space + FPN+
# dataset: cityscapes
# image scale: 0.375 --> 384 x 768
python search/search.py --search-space basic \
  --supernet-weights pretrained/ofa_fanet/basic_plus/ofa_plus_basic_depth-expand-width@phase2_1e-2/model_final.pth \
  --meta-data data/ofa_fanet_plus_basic_rtx_fps@0.375.json --dataset cityscapes \
  --data-root /home/cseadmin/huangsh/datasets/Cityscapes --scale 0.375 --sec-obj latency --surrogate lgb --save-path tmp

# search space: basic block search space + FPN+
# dataset: cityscapes
# image scale: 0.5 --> 512 x 1024
python search/search.py --search-space basic \
  --supernet-weights pretrained/ofa_fanet/basic_plus/ofa_plus_basic_depth-expand-width@phase2_1e-2/model_final.pth \
  --meta-data data/ofa_fanet_plus_basic_rtx_fps@0.5.json --dataset cityscapes \
  --data-root /home/cseadmin/huangsh/datasets/Cityscapes --scale 0.5 --sec-obj latency --surrogate lgb --save-path tmp

# search space: basic block search space + FPN+
# dataset: cityscapes
# image scale: 0.625 --> 640 x 1280
python search/search.py --search-space basic \
  --supernet-weights pretrained/ofa_fanet/basic_plus/ofa_plus_basic_depth-expand-width@phase2_1e-2/model_final.pth \
  --meta-data data/ofa_fanet_plus_basic_rtx_fps@0.625.json --dataset cityscapes \
  --data-root /home/cseadmin/huangsh/datasets/Cityscapes --scale 0.625 --sec-obj latency --surrogate lgb --save-path tmp

# search space: basic block search space + FPN+
# dataset: cityscapes
# image scale: 0.75 --> 768 x 1536
python search/search.py --search-space basic \
  --supernet-weights pretrained/ofa_fanet/basic_plus/ofa_plus_basic_depth-expand-width@phase2_1e-2/model_final.pth \
  --meta-data data/ofa_fanet_plus_basic_rtx_fps@0.75.json --dataset cityscapes \
  --data-root /home/cseadmin/huangsh/datasets/Cityscapes --scale 0.75 --sec-obj latency --surrogate lgb --save-path tmp