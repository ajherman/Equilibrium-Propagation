#!/bin/sh
python ../main.py \
    --model 'autoLCACNN' \
    --alg 'LCA' \
    --task 'CIFAR10' \
    --channels 64 \
    --kernels 6 \
    --pools 'iamm' \
    --strides 1 1 \
    --paddings 3 1 \
    --sparse-layers 0 \
    --lambdas 0.2 \
    --fc 10 \
    --optim 'sgd' \
    --mmt 0.9 \
    --lrs 2.5e-1 0.0 \
    --lr-decay \
    --wds 3e-4 3e-4 3e-4 3e-4 3e-4 3e-4 \
    --act 'relu' \
    --todo 'train' \
    --betas 0.0 0.5 \
    --T1 800 \
    --T2 0 \
    --mbs 128 \
    --loss 'mse' \
    --data-aug \
    --save  \
    --device $2 \
    --seed $1 \
    --epochs 120 \
    --tensorboard \
    --dt 0.01 \
    #--competitiontype "feature_inner_products" \
    #--inhibitstrength 1.0 \
    #--comp-syn-constraints "rowunitnorm" \
