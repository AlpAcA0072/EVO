#!/bin/bash
NUM_PROC=$1
shift
#python -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port=$RANDOM abl_train_imagenet.py "$@"
python -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port=2245 train_imagenet.py "$@"

# training for bottleneck ofa backbone
# sh ./scripts/distributed_train_imagenet.sh 2 /imagenet/ --teacher resnet50d
# --lr 0.1 --warmup-epochs 5 --epochs 220 --weight-decay 1e-4 --sched cosine --reprob 0.4
# --remode pixel -b 112 --apex -j 12

# training for inverted bottleneck ofa backbone
# sh ./scripts/distributed_train_imagenet.sh 8 /imagenet/ --teacher mobilenetv3_large_100
# -b 128 --sched step --epochs 450 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf
# --opt-eps .001 -j 12 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.3 --drop-connect 0.2
# --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --amp --lr .048 --apex