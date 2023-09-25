#!/bin/sh
python ../main.py \
    --model 'autoLCACNN' \
    --alg 'LCA' \
    --task 'CIFAR10' \
    --channels 64 128 256 256 \
    --kernels 8 4 4 4 \
    --pools 'mmmm' \
    --strides 1 1 1 1 \
    --paddings 4 1 1 1 \
    --sparse-layers 0 1 2 3 4 \
    --lambdas 0.1 0.05 0.005 0.005 0.005 \
    --fc 10 \
    --optim 'sgd' \
    --mmt 0.9 \
    --lrs 2.5e-1 1.0e-1 5.0e-2 3.0e-2 1.0e-2 \
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
    --load-path-convert results/LCA/mse/2023-09-20/11-48-27_gpu0 \
    --convert-place-layers 0 - - - \
    #--competitiontype "feature_inner_products" \
    #--inhibitstrength 1.0 \
    #--comp-syn-constraints "rowunitnorm" \
