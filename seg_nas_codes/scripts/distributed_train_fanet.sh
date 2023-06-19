#!/bin/bash

## train basic ofa-fanet-plus on full capacity first
#python -m torch.distributed.launch --nproc_per_node=8 train/train_fanet.py --data /defaultShare/Cityscapes/ --nccl_ip 12 --n_img_per_gpu 6 --backbone pretrained/backbone/basic.pth --kd

# train bottleneck ofa-fanet-plus on full capacity first
#python -m torch.distributed.launch --nproc_per_node=2 train/train_fanet.py --data /home/cseadmin/datasets/Cityscapes/ --nccl_ip 12 --n_img_per_gpu 24 --n_workers_train 20 --backbone pretrained/backbone/bottleneck.pth --kd

# train inverted bottleneck ofa-fanet on full capacity first
#python -m torch.distributed.launch --nproc_per_node=8 train/train_fanet.py --data /defaultShare/Cityscapes/ --nccl_ip 12 --n_img_per_gpu 6 --backbone pretrained/backbone/inverted.pth --kd

# train r152d based FANetPlus
python -m torch.distributed.launch --nproc_per_node=6 train/train_fanet.py --data /defaultShare/Cityscapes/ --nccl_ip 12 --n_img_per_gpu 8
