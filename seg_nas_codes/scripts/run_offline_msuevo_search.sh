#!/bin/bash

# search space: basic block search space + FPN+
# dataset: cityscapes
# image scale: 0.5 --> 512 x 1024
# objectives: mIoU + latency
python search/offline_search.py --search-space basic \
  --meta-data data/ofa_fanet_plus_basic_rtx_fps@0.5.json --sec-obj latency --surrogate lgb \
  --save-path OfflineSearch
