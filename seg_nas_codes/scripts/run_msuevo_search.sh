#!/bin/bash

## search space: basic block search space + FaPN
## dataset: cityscapes
## objectives: mIoU + latency
#python search/search.py --search-space basic \
#  --supernet-weights pretrained/ofa_fanet/basic/ofa_basic_depth-expand-width@phase2/model_maxmIOU50.pth \
#  --meta-data data/ofa_fanet_basic_rtx_fps@0.5.json --dataset cityscapes \
#  --data-root /home/cseadmin/huangsh/datasets/Cityscapes --sec-obj latency --surrogate lgb --save-path tmp

# search space: basic block search space + FPN+
# dataset: cityscapes
# image scale: 0.375 --> 384 x 768
# objectives: mIoU + latency
#python search/search.py --search-space basic \
#  --supernet-weights pretrained/ofa_fanet/basic_plus/ofa_plus_basic_depth-expand-width@phase2_1e-2/model_final.pth \
#  --meta-data data/RSearch-cityscapes-BasicSearchSpace-scale@0.375.json --dataset cityscapes \
#  --data-root /home/cseadmin/huangsh/datasets/Cityscapes --scale 0.375 --sec-obj latency --surrogate lgb --save-path scale@0.375

## search space: basic block search space + FPN+
## dataset: cityscapes
## image scale: 0.5 --> 512 x 1024
## objectives: mIoU + latency
#python search/search.py --search-space basic \
#  --supernet-weights pretrained/ofa_fanet/basic_plus/ofa_plus_basic_depth-expand-width@phase2_1e-2/model_final.pth \
#  --dataset cityscapes --data-root /home/cseadmin/huangsh/datasets/Cityscapes --scale 0.5 --sec-obj latency \
#  --surrogate lgb --save-path SearchExps

# search space: basic block search space + FPN+
# dataset: CamVid
# image scale: 1.0 --> 720 x 960
# objectives: mIoU + latency
python search/search.py --search-space basic \
  --supernet-weights pretrained/ofa_fanet/camvid/ofa_plus_basic_camvid_depth-expand-width@phase2/model_maxmIOUFull.pth \
  --dataset camvid --data-root /home/cseadmin/huangsh/datasets/CamVid --scale 1.0 --sec-obj latency \
  --surrogate lgb --save-path SearchExps

## search space: basic block search space + FPN+
## dataset: cityscapes
## image scale: 0.625 --> 640 x 1280
## objectives: mIoU + latency
#python search/search.py --search-space basic \
#  --supernet-weights pretrained/ofa_fanet/basic_plus/ofa_plus_basic_depth-expand-width@phase2_1e-2/model_final.pth \
#  --meta-data data/ofa_fanet_plus_basic_rtx_fps@0.625.json --dataset cityscapes \
#  --data-root /home/cseadmin/huangsh/datasets/Cityscapes --scale 0.625 --sec-obj latency --surrogate lgb --save-path scale@0.625
#
## search space: basic block search space + FPN+
## dataset: cityscapes
## image scale: 0.75 --> 768 x 1536
## objectives: mIoU + latency
#python search/search.py --search-space basic \
#  --supernet-weights pretrained/ofa_fanet/basic_plus/ofa_plus_basic_depth-expand-width@phase2_1e-2/model_final.pth \
#  --meta-data data/ofa_fanet_plus_basic_rtx_fps@0.75.json --dataset cityscapes \
#  --data-root /home/cseadmin/huangsh/datasets/Cityscapes --scale 0.75 --sec-obj latency --surrogate lgb --save-path scale@0.75