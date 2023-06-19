#!/bin/bash

# basic block search space
python latency.py --sample-size 1200 --backbone basic --size_hw 384 768 --save ofa_fanet_plus_basic_rtx_fps@0.375.json
wait
python latency.py --sample-size 1200 --backbone basic --size_hw 512 1024 --save ofa_fanet_plus_basic_rtx_fps@0.5.json
wait
python latency.py --sample-size 1200 --backbone basic --size_hw 640 1280 --save ofa_fanet_plus_basic_rtx_fps@0.625.json
wait
python latency.py --sample-size 1200 --backbone basic --size_hw 768 1536 --save ofa_fanet_plus_basic_rtx_fps@0.75.json
wait

## bottleneck block search space
#python latency.py --sample-size 12000 --backbone bottleneck --save ofa_fanet_bottleneck_rtx_fps@0.5.json
#wait
#
## inverted bottleneck block search space
#python latency.py --sample-size 12000 --backbone inverted --save ofa_fanet_inverted_rtx_fps@0.5.json
#wait