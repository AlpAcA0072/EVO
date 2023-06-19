#!/bin/bash

python -m torch.distributed.launch --nproc_per_node=2 train/finetune_subnet.py \
--data /home/cseadmin/datasets/Cityscapes/ --nccl_ip 12 --n_img_per_gpu 24 --backbone basic \
--ckpt ./pretrained/ofa_fanet/basic_plus/ofa_plus_basic_depth-expand-width@phase2_1e-2/model_final.pth \
--subnet Exps/BasicSearchSpace-mIoU&latency-lgb-n_doe@100-n_iter@8-max_iter@30/high_tradeoff_subnets/subnet_9.json --kd

python -m torch.distributed.launch --nproc_per_node=2 train/finetune_subnet.py \
--data /home/cseadmin/datasets/Cityscapes/ --nccl_ip 12 --n_img_per_gpu 24 --backbone basic \
--ckpt ./pretrained/ofa_fanet/basic_plus/ofa_plus_basic_depth-expand-width@phase2_1e-2/model_final.pth \
--subnet Exps/BasicSearchSpace-mIoU&latency-lgb-n_doe@100-n_iter@8-max_iter@30/high_tradeoff_subnets/subnet_26.json --kd

python -m torch.distributed.launch --nproc_per_node=2 train/finetune_subnet.py \
--data /home/cseadmin/datasets/Cityscapes/ --nccl_ip 12 --n_img_per_gpu 24 --backbone basic \
--ckpt ./pretrained/ofa_fanet/basic_plus/ofa_plus_basic_depth-expand-width@phase2_1e-2/model_final.pth \
--subnet Exps/BasicSearchSpace-mIoU&latency-lgb-n_doe@100-n_iter@8-max_iter@30/high_tradeoff_subnets/subnet_28.json --kd

python -m torch.distributed.launch --nproc_per_node=2 train/finetune_subnet.py \
--data /home/cseadmin/datasets/Cityscapes/ --nccl_ip 12 --n_img_per_gpu 24 --backbone basic \
--ckpt ./pretrained/ofa_fanet/basic_plus/ofa_plus_basic_depth-expand-width@phase2_1e-2/model_final.pth \
--subnet Exps/BasicSearchSpace-mIoU&latency-lgb-n_doe@100-n_iter@8-max_iter@30/high_tradeoff_subnets/subnet_29.json --kd

python -m torch.distributed.launch --nproc_per_node=2 train/finetune_subnet.py \
--data /home/cseadmin/datasets/Cityscapes/ --nccl_ip 12 --n_img_per_gpu 24 --backbone basic \
--ckpt ./pretrained/ofa_fanet/basic_plus/ofa_plus_basic_depth-expand-width@phase2_1e-2/model_final.pth \
--subnet Exps/BasicSearchSpace-mIoU&latency-lgb-n_doe@100-n_iter@8-max_iter@30/high_tradeoff_subnets/subnet_34.json --kd

python -m torch.distributed.launch --nproc_per_node=2 train/finetune_subnet.py \
--data /home/cseadmin/datasets/Cityscapes/ --nccl_ip 12 --n_img_per_gpu 24 --backbone basic \
--ckpt ./pretrained/ofa_fanet/basic_plus/ofa_plus_basic_depth-expand-width@phase2_1e-2/model_final.pth \
--subnet Exps/BasicSearchSpace-mIoU&latency-lgb-n_doe@100-n_iter@8-max_iter@30/high_tradeoff_subnets/subnet_40.json --kd