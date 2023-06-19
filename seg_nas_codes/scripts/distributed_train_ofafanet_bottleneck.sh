#!/bin/bash

## depth phase 1
#python -m torch.distributed.launch --nproc_per_node=2 train/train_ofafanet_bottleneck.py --data /home/cseadmin/datasets/Cityscapes/ --nccl_ip 10 --n_img_per_gpu 24 --ckpt pretrained/ofa_fanet/bottleneck_plus/ofa_plus_bottleneck.pth
#wait
## depth phase 2
#python -m torch.distributed.launch --nproc_per_node=2 train/train_ofafanet_bottleneck.py --data /home/cseadmin/datasets/Cityscapes/ --nccl_ip 11 --n_img_per_gpu 24 --ckpt ../Exps/ofa_plus_bottleneck_depth@phase1/model_maxmIOU50.pth --phase 2
#wait
## depth+expand phase 1
#python -m torch.distributed.launch --nproc_per_node=2 train/train_ofafanet_bottleneck.py --data /home/cseadmin/datasets/Cityscapes/ --nccl_ip 12 --n_img_per_gpu 24 --ckpt ../Exps/ofa_plus_bottleneck_depth@phase2/model_maxmIOU50.pth --task expand
#wait
## depth+expand phase 2
#python -m torch.distributed.launch --nproc_per_node=2 train/train_ofafanet_bottleneck.py --data /home/cseadmin/datasets/Cityscapes/ --nccl_ip 13 --n_img_per_gpu 24 --ckpt ../Exps/ofa_plus_bottleneck_depth-expand@phase1/model_maxmIOU50.pth --task expand --phase 2
#wait
# depth+expand+width phase 1
python -m torch.distributed.launch --nproc_per_node=2 train/train_ofafanet_bottleneck.py --data /home/cseadmin/datasets/Cityscapes/ --nccl_ip 14 --n_img_per_gpu 24 --ckpt ../Exps/ofa_plus_bottleneck_depth-expand@phase2/model_maxmIOU50.pth --task width
wait
# depth+expand+width phase 2
python -m torch.distributed.launch --nproc_per_node=2 train/train_ofafanet_bottleneck.py --data /home/cseadmin/datasets/Cityscapes/ --nccl_ip 15 --n_img_per_gpu 24 --ckpt ../Exps/ofa_plus_bottleneck_depth-expand-width@phase1/model_maxmIOU50.pth --task width --phase 2
wait