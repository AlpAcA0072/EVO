#!/bin/bash

python evaluation/multi_scale_evaluation.py --data ~/huangsh/datasets/Cityscapes/ \
--backbone basic --ckpt pretrained/ofa_fanet/basic_plus/ofa_plus_basic_depth-expand-width@phase2_1e-2/model_final.pth \
--log logs --subnet Exps/BasicSearchSpace-mIoU\&latency-lgb-n_doe@100-n_iter@8-max_iter@30/non_dominated_subnets/subnet_1.json

python evaluation/multi_scale_evaluation.py --data ~/huangsh/datasets/Cityscapes/ \
--backbone basic --ckpt pretrained/ofa_fanet/basic_plus/ofa_plus_basic_depth-expand-width@phase2_1e-2/model_final.pth \
--log logs --subnet Exps/BasicSearchSpace-mIoU\&latency-lgb-n_doe@100-n_iter@8-max_iter@30/non_dominated_subnets/subnet_2.json

python evaluation/multi_scale_evaluation.py --data ~/huangsh/datasets/Cityscapes/ \
--backbone basic --ckpt pretrained/ofa_fanet/basic_plus/ofa_plus_basic_depth-expand-width@phase2_1e-2/model_final.pth \
--log logs --subnet Exps/BasicSearchSpace-mIoU\&latency-lgb-n_doe@100-n_iter@8-max_iter@30/non_dominated_subnets/subnet_3.json

python evaluation/multi_scale_evaluation.py --data ~/huangsh/datasets/Cityscapes/ \
--backbone basic --ckpt pretrained/ofa_fanet/basic_plus/ofa_plus_basic_depth-expand-width@phase2_1e-2/model_final.pth \
--log logs --subnet Exps/BasicSearchSpace-mIoU\&latency-lgb-n_doe@100-n_iter@8-max_iter@30/non_dominated_subnets/subnet_4.json

python evaluation/multi_scale_evaluation.py --data ~/huangsh/datasets/Cityscapes/ \
--backbone basic --ckpt pretrained/ofa_fanet/basic_plus/ofa_plus_basic_depth-expand-width@phase2_1e-2/model_final.pth \
--log logs --subnet Exps/BasicSearchSpace-mIoU\&latency-lgb-n_doe@100-n_iter@8-max_iter@30/non_dominated_subnets/subnet_5.json

python evaluation/multi_scale_evaluation.py --data ~/huangsh/datasets/Cityscapes/ \
--backbone basic --ckpt pretrained/ofa_fanet/basic_plus/ofa_plus_basic_depth-expand-width@phase2_1e-2/model_final.pth \
--log logs --subnet Exps/BasicSearchSpace-mIoU\&latency-lgb-n_doe@100-n_iter@8-max_iter@30/non_dominated_subnets/subnet_6.json

python evaluation/multi_scale_evaluation.py --data ~/huangsh/datasets/Cityscapes/ \
--backbone basic --ckpt pretrained/ofa_fanet/basic_plus/ofa_plus_basic_depth-expand-width@phase2_1e-2/model_final.pth \
--log logs --subnet Exps/BasicSearchSpace-mIoU\&latency-lgb-n_doe@100-n_iter@8-max_iter@30/non_dominated_subnets/subnet_7.json

python evaluation/multi_scale_evaluation.py --data ~/huangsh/datasets/Cityscapes/ \
--backbone basic --ckpt pretrained/ofa_fanet/basic_plus/ofa_plus_basic_depth-expand-width@phase2_1e-2/model_final.pth \
--log logs --subnet Exps/BasicSearchSpace-mIoU\&latency-lgb-n_doe@100-n_iter@8-max_iter@30/non_dominated_subnets/subnet_8.json

python evaluation/multi_scale_evaluation.py --data ~/huangsh/datasets/Cityscapes/ \
--backbone basic --ckpt pretrained/ofa_fanet/basic_plus/ofa_plus_basic_depth-expand-width@phase2_1e-2/model_final.pth \
--log logs --subnet Exps/BasicSearchSpace-mIoU\&latency-lgb-n_doe@100-n_iter@8-max_iter@30/non_dominated_subnets/subnet_9.json

python evaluation/multi_scale_evaluation.py --data ~/huangsh/datasets/Cityscapes/ \
--backbone basic --ckpt pretrained/ofa_fanet/basic_plus/ofa_plus_basic_depth-expand-width@phase2_1e-2/model_final.pth \
--log logs --subnet Exps/BasicSearchSpace-mIoU\&latency-lgb-n_doe@100-n_iter@8-max_iter@30/non_dominated_subnets/subnet_10.json

python evaluation/multi_scale_evaluation.py --data ~/huangsh/datasets/Cityscapes/ \
--backbone basic --ckpt pretrained/ofa_fanet/basic_plus/ofa_plus_basic_depth-expand-width@phase2_1e-2/model_final.pth \
--log logs --subnet Exps/BasicSearchSpace-mIoU\&latency-lgb-n_doe@100-n_iter@8-max_iter@30/non_dominated_subnets/subnet_11.json

python evaluation/multi_scale_evaluation.py --data ~/huangsh/datasets/Cityscapes/ \
--backbone basic --ckpt pretrained/ofa_fanet/basic_plus/ofa_plus_basic_depth-expand-width@phase2_1e-2/model_final.pth \
--log logs --subnet Exps/BasicSearchSpace-mIoU\&latency-lgb-n_doe@100-n_iter@8-max_iter@30/non_dominated_subnets/subnet_12.json

python evaluation/multi_scale_evaluation.py --data ~/huangsh/datasets/Cityscapes/ \
--backbone basic --ckpt pretrained/ofa_fanet/basic_plus/ofa_plus_basic_depth-expand-width@phase2_1e-2/model_final.pth \
--log logs --subnet Exps/BasicSearchSpace-mIoU\&latency-lgb-n_doe@100-n_iter@8-max_iter@30/non_dominated_subnets/subnet_13.json

python evaluation/multi_scale_evaluation.py --data ~/huangsh/datasets/Cityscapes/ \
--backbone basic --ckpt pretrained/ofa_fanet/basic_plus/ofa_plus_basic_depth-expand-width@phase2_1e-2/model_final.pth \
--log logs --subnet Exps/BasicSearchSpace-mIoU\&latency-lgb-n_doe@100-n_iter@8-max_iter@30/non_dominated_subnets/subnet_14.json

python evaluation/multi_scale_evaluation.py --data ~/huangsh/datasets/Cityscapes/ \
--backbone basic --ckpt pretrained/ofa_fanet/basic_plus/ofa_plus_basic_depth-expand-width@phase2_1e-2/model_final.pth \
--log logs --subnet Exps/BasicSearchSpace-mIoU\&latency-lgb-n_doe@100-n_iter@8-max_iter@30/non_dominated_subnets/subnet_15.json

python evaluation/multi_scale_evaluation.py --data ~/huangsh/datasets/Cityscapes/ \
--backbone basic --ckpt pretrained/ofa_fanet/basic_plus/ofa_plus_basic_depth-expand-width@phase2_1e-2/model_final.pth \
--log logs --subnet Exps/BasicSearchSpace-mIoU\&latency-lgb-n_doe@100-n_iter@8-max_iter@30/non_dominated_subnets/subnet_16.json

python evaluation/multi_scale_evaluation.py --data ~/huangsh/datasets/Cityscapes/ \
--backbone basic --ckpt pretrained/ofa_fanet/basic_plus/ofa_plus_basic_depth-expand-width@phase2_1e-2/model_final.pth \
--log logs --subnet Exps/BasicSearchSpace-mIoU\&latency-lgb-n_doe@100-n_iter@8-max_iter@30/non_dominated_subnets/subnet_17.json

python evaluation/multi_scale_evaluation.py --data ~/huangsh/datasets/Cityscapes/ \
--backbone basic --ckpt pretrained/ofa_fanet/basic_plus/ofa_plus_basic_depth-expand-width@phase2_1e-2/model_final.pth \
--log logs --subnet Exps/BasicSearchSpace-mIoU\&latency-lgb-n_doe@100-n_iter@8-max_iter@30/non_dominated_subnets/subnet_18.json

python evaluation/multi_scale_evaluation.py --data ~/huangsh/datasets/Cityscapes/ \
--backbone basic --ckpt pretrained/ofa_fanet/basic_plus/ofa_plus_basic_depth-expand-width@phase2_1e-2/model_final.pth \
--log logs --subnet Exps/BasicSearchSpace-mIoU\&latency-lgb-n_doe@100-n_iter@8-max_iter@30/non_dominated_subnets/subnet_19.json

python evaluation/multi_scale_evaluation.py --data ~/huangsh/datasets/Cityscapes/ \
--backbone basic --ckpt pretrained/ofa_fanet/basic_plus/ofa_plus_basic_depth-expand-width@phase2_1e-2/model_final.pth \
--log logs --subnet Exps/BasicSearchSpace-mIoU\&latency-lgb-n_doe@100-n_iter@8-max_iter@30/non_dominated_subnets/subnet_20.json

python evaluation/multi_scale_evaluation.py --data ~/huangsh/datasets/Cityscapes/ \
--backbone basic --ckpt pretrained/ofa_fanet/basic_plus/ofa_plus_basic_depth-expand-width@phase2_1e-2/model_final.pth \
--log logs --subnet Exps/BasicSearchSpace-mIoU\&latency-lgb-n_doe@100-n_iter@8-max_iter@30/non_dominated_subnets/subnet_21.json

python evaluation/multi_scale_evaluation.py --data ~/huangsh/datasets/Cityscapes/ \
--backbone basic --ckpt pretrained/ofa_fanet/basic_plus/ofa_plus_basic_depth-expand-width@phase2_1e-2/model_final.pth \
--log logs --subnet Exps/BasicSearchSpace-mIoU\&latency-lgb-n_doe@100-n_iter@8-max_iter@30/non_dominated_subnets/subnet_22.json

python evaluation/multi_scale_evaluation.py --data ~/huangsh/datasets/Cityscapes/ \
--backbone basic --ckpt pretrained/ofa_fanet/basic_plus/ofa_plus_basic_depth-expand-width@phase2_1e-2/model_final.pth \
--log logs --subnet Exps/BasicSearchSpace-mIoU\&latency-lgb-n_doe@100-n_iter@8-max_iter@30/non_dominated_subnets/subnet_23.json

python evaluation/multi_scale_evaluation.py --data ~/huangsh/datasets/Cityscapes/ \
--backbone basic --ckpt pretrained/ofa_fanet/basic_plus/ofa_plus_basic_depth-expand-width@phase2_1e-2/model_final.pth \
--log logs --subnet Exps/BasicSearchSpace-mIoU\&latency-lgb-n_doe@100-n_iter@8-max_iter@30/non_dominated_subnets/subnet_24.json

python evaluation/multi_scale_evaluation.py --data ~/huangsh/datasets/Cityscapes/ \
--backbone basic --ckpt pretrained/ofa_fanet/basic_plus/ofa_plus_basic_depth-expand-width@phase2_1e-2/model_final.pth \
--log logs --subnet Exps/BasicSearchSpace-mIoU\&latency-lgb-n_doe@100-n_iter@8-max_iter@30/non_dominated_subnets/subnet_25.json

python evaluation/multi_scale_evaluation.py --data ~/huangsh/datasets/Cityscapes/ \
--backbone basic --ckpt pretrained/ofa_fanet/basic_plus/ofa_plus_basic_depth-expand-width@phase2_1e-2/model_final.pth \
--log logs --subnet Exps/BasicSearchSpace-mIoU\&latency-lgb-n_doe@100-n_iter@8-max_iter@30/non_dominated_subnets/subnet_26.json

python evaluation/multi_scale_evaluation.py --data ~/huangsh/datasets/Cityscapes/ \
--backbone basic --ckpt pretrained/ofa_fanet/basic_plus/ofa_plus_basic_depth-expand-width@phase2_1e-2/model_final.pth \
--log logs --subnet Exps/BasicSearchSpace-mIoU\&latency-lgb-n_doe@100-n_iter@8-max_iter@30/non_dominated_subnets/subnet_27.json

python evaluation/multi_scale_evaluation.py --data ~/huangsh/datasets/Cityscapes/ \
--backbone basic --ckpt pretrained/ofa_fanet/basic_plus/ofa_plus_basic_depth-expand-width@phase2_1e-2/model_final.pth \
--log logs --subnet Exps/BasicSearchSpace-mIoU\&latency-lgb-n_doe@100-n_iter@8-max_iter@30/non_dominated_subnets/subnet_28.json

python evaluation/multi_scale_evaluation.py --data ~/huangsh/datasets/Cityscapes/ \
--backbone basic --ckpt pretrained/ofa_fanet/basic_plus/ofa_plus_basic_depth-expand-width@phase2_1e-2/model_final.pth \
--log logs --subnet Exps/BasicSearchSpace-mIoU\&latency-lgb-n_doe@100-n_iter@8-max_iter@30/non_dominated_subnets/subnet_29.json

python evaluation/multi_scale_evaluation.py --data ~/huangsh/datasets/Cityscapes/ \
--backbone basic --ckpt pretrained/ofa_fanet/basic_plus/ofa_plus_basic_depth-expand-width@phase2_1e-2/model_final.pth \
--log logs --subnet Exps/BasicSearchSpace-mIoU\&latency-lgb-n_doe@100-n_iter@8-max_iter@30/non_dominated_subnets/subnet_30.json

python evaluation/multi_scale_evaluation.py --data ~/huangsh/datasets/Cityscapes/ \
--backbone basic --ckpt pretrained/ofa_fanet/basic_plus/ofa_plus_basic_depth-expand-width@phase2_1e-2/model_final.pth \
--log logs --subnet Exps/BasicSearchSpace-mIoU\&latency-lgb-n_doe@100-n_iter@8-max_iter@30/non_dominated_subnets/subnet_31.json

python evaluation/multi_scale_evaluation.py --data ~/huangsh/datasets/Cityscapes/ \
--backbone basic --ckpt pretrained/ofa_fanet/basic_plus/ofa_plus_basic_depth-expand-width@phase2_1e-2/model_final.pth \
--log logs --subnet Exps/BasicSearchSpace-mIoU\&latency-lgb-n_doe@100-n_iter@8-max_iter@30/non_dominated_subnets/subnet_32.json

python evaluation/multi_scale_evaluation.py --data ~/huangsh/datasets/Cityscapes/ \
--backbone basic --ckpt pretrained/ofa_fanet/basic_plus/ofa_plus_basic_depth-expand-width@phase2_1e-2/model_final.pth \
--log logs --subnet Exps/BasicSearchSpace-mIoU\&latency-lgb-n_doe@100-n_iter@8-max_iter@30/non_dominated_subnets/subnet_33.json

python evaluation/multi_scale_evaluation.py --data ~/huangsh/datasets/Cityscapes/ \
--backbone basic --ckpt pretrained/ofa_fanet/basic_plus/ofa_plus_basic_depth-expand-width@phase2_1e-2/model_final.pth \
--log logs --subnet Exps/BasicSearchSpace-mIoU\&latency-lgb-n_doe@100-n_iter@8-max_iter@30/non_dominated_subnets/subnet_34.json

python evaluation/multi_scale_evaluation.py --data ~/huangsh/datasets/Cityscapes/ \
--backbone basic --ckpt pretrained/ofa_fanet/basic_plus/ofa_plus_basic_depth-expand-width@phase2_1e-2/model_final.pth \
--log logs --subnet Exps/BasicSearchSpace-mIoU\&latency-lgb-n_doe@100-n_iter@8-max_iter@30/non_dominated_subnets/subnet_35.json

python evaluation/multi_scale_evaluation.py --data ~/huangsh/datasets/Cityscapes/ \
--backbone basic --ckpt pretrained/ofa_fanet/basic_plus/ofa_plus_basic_depth-expand-width@phase2_1e-2/model_final.pth \
--log logs --subnet Exps/BasicSearchSpace-mIoU\&latency-lgb-n_doe@100-n_iter@8-max_iter@30/non_dominated_subnets/subnet_36.json

python evaluation/multi_scale_evaluation.py --data ~/huangsh/datasets/Cityscapes/ \
--backbone basic --ckpt pretrained/ofa_fanet/basic_plus/ofa_plus_basic_depth-expand-width@phase2_1e-2/model_final.pth \
--log logs --subnet Exps/BasicSearchSpace-mIoU\&latency-lgb-n_doe@100-n_iter@8-max_iter@30/non_dominated_subnets/subnet_37.json

python evaluation/multi_scale_evaluation.py --data ~/huangsh/datasets/Cityscapes/ \
--backbone basic --ckpt pretrained/ofa_fanet/basic_plus/ofa_plus_basic_depth-expand-width@phase2_1e-2/model_final.pth \
--log logs --subnet Exps/BasicSearchSpace-mIoU\&latency-lgb-n_doe@100-n_iter@8-max_iter@30/non_dominated_subnets/subnet_38.json

python evaluation/multi_scale_evaluation.py --data ~/huangsh/datasets/Cityscapes/ \
--backbone basic --ckpt pretrained/ofa_fanet/basic_plus/ofa_plus_basic_depth-expand-width@phase2_1e-2/model_final.pth \
--log logs --subnet Exps/BasicSearchSpace-mIoU\&latency-lgb-n_doe@100-n_iter@8-max_iter@30/non_dominated_subnets/subnet_39.json

python evaluation/multi_scale_evaluation.py --data ~/huangsh/datasets/Cityscapes/ \
--backbone basic --ckpt pretrained/ofa_fanet/basic_plus/ofa_plus_basic_depth-expand-width@phase2_1e-2/model_final.pth \
--log logs --subnet Exps/BasicSearchSpace-mIoU\&latency-lgb-n_doe@100-n_iter@8-max_iter@30/non_dominated_subnets/subnet_40.json