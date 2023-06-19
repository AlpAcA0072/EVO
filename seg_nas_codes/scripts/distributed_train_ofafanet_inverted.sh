#!/bin/bash

## depth phase 1
#python -m torch.distributed.launch --nproc_per_node=2 train/train_ofafanet_inverted.py --data /home/cseadmin/datasets/Cityscapes/ --nccl_ip 10 --n_img_per_gpu 24 --ckpt pretrained/ofa_fanet/inverted_plus/ofa_plus_inverted.pth
#wait
## depth phase 2
#python -m torch.distributed.launch --nproc_per_node=2 train/train_ofafanet_inverted.py --data /home/cseadmin/datasets/Cityscapes/ --nccl_ip 11 --n_img_per_gpu 24 --ckpt ../Exps/ofa_plus_inverted_depth@phase1/model_maxmIOU50.pth --phase 2
#wait
# depth+expand phase 1
python -m torch.distributed.launch --nproc_per_node=2 train/train_ofafanet_inverted.py --data /home/cseadmin/datasets/Cityscapes/ --nccl_ip 12 --n_img_per_gpu 24 --ckpt ../Exps/ofa_plus_inverted_depth@phase2/model_maxmIOU50.pth --task expand
wait
# depth+expand phase 2
python -m torch.distributed.launch --nproc_per_node=2 train/train_ofafanet_inverted.py --data /home/cseadmin/datasets/Cityscapes/ --nccl_ip 13 --n_img_per_gpu 24 --ckpt ../Exps/ofa_plus_inverted_depth-expand@phase1/model_maxmIOU50.pth --task expand --phase 2
wait