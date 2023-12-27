#!/bin/sh
python ../main.py \
    --model 'CNN' \
    --task 'CIFAR10' \
    --channels 128 256 512 512 \
    --kernels 8 4 3 3 \
    --pools 'mmmm' \
    --strides 1 1 1 1 \
    --paddings 4 2 1 0 \
    --fc 10 \
    --optim 'sgd' \
    --mmt 0.9 \
    --lrs 0.25 0.15 0.1 0.08 0.05 \
    --lr-decay \
    --epochs 120 \
    --wds 3e-4 3e-4 3e-4 3e-4 3e-4 \
    --act 'my_hard_sig' \
    --todo 'train' \
    --betas 0.0 0.5 \
    --thirdphase \
    --T1 250 \
    --T2 25 \
    --mbs 128 \
    --loss 'mse' \
    --data-aug \
    --save  \
    --device $2 \
    --seed $1 \
#    --alg CEP \
#    --load-path results/EP/cel/2023-07-18/12-05-15_gpu0

