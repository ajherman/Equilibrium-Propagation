#!/bin/sh
python ../main.py \
    --model 'HopfieldCNN' \
    --task 'CIFAR10' \
    --channels 128 256 512 512 \
    --kernels 3 3 3 3 \
    --pools 'mmmm' \
    --strides 1 1 1 1 \
    --paddings 1 1 1 0 \
    --fc 10 \
    --lat-layers -1 \
    --lat-lrs 0.01 \
    --optim 'sgd' \
    --mmt 0.9 \
    --lrs 0.25 0.15 0.1 0.08 0.05 \
    --lr-decay \
    --epochs 120 \
    --wds 3e-4 3e-4 3e-4 3e-4 3e-4 \
    --act 'mysig' \
    --todo 'train' \
    --betas 0.0 0.5 \
    --thirdphase \
    --T1 250 \
    --T2 25 \
    --mbs 128 \
    --loss 'mse' \
    --save  \
    --device 0 \
    --seed 3009345 \
    --load-path results/EP/mse/2023-07-24/23-13-24_gpu0 \
#    --alg CEP \
#    --load-path results/EP/cel/2023-07-18/12-05-15_gpu0

