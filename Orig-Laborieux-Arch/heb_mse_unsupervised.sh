#!/bin/sh
python ../main.py \
    --model 'CNN' \
    --task 'CIFAR10' \
    --channels 128 256 512 512 \
    --kernels 3 3 3 3 \
    --pools 'mmmm' \
    --strides 1 1 1 1 \
    --paddings 1 1 1 0 \
    --fc 10 \
    --optim 'sgd' \
    --mmt 0.9 \
    --lrs 0.25 0.15 0.1 0.08 0.05 \
    --lr-decay \
    --epochs 2 \
    --wds 3e-4 3e-4 3e-4 3e-4 3e-4 \
    --act 'my_hard_sig' \
    --todo 'train' \
    --T1 250 \
    --mbs 128 \
    --loss 'mse' \
    --data-aug \
    --seed $1 \
    --save  \
    --device 0 \
    --alg HebUnsupervised \

