#!/bin/sh
for lambda in 0.0 0.05 0.1 0.15 0.20 ; do
    echo "now running with lambda=$lambda";
    python ../main.py \
        --model 'autoLCACNN' \
        --alg 'LCAEP' \
        --thirdphase \
        --task 'CIFAR10' \
        --channels 64 128 256 256 \
        --kernels 6 3 3 3 \
        --lrs 2.5e-2 1.5e-2 1.0e-2 8.0e-3  5.0e-3 \
        --lr-decay \
        --wds 3e-4 3e-4 3e-4 3e-4 3e-4 3e-4 \
        --pools 'mmmm' \
        --strides 1 1 1 1 \
        --paddings 3 1 1 0 \
        --sparse-layers 0 \
        --lambdas $lambda \
        --fc 10 \
        --optim 'sgd' \
        --mmt 0.9 \
        --act 'my_hard_sig' \
        --todo 'train' \
        --betas 0.0 0.5 \
        --T1 600 \
        --T2 50 \
        --mbs 128 \
        --data-aug \
        --save  \
        --device $2 \
        --seed $1 \
        --tensorboard \
        --dt 0.01 \
        --load-path-convert results/EP/mse/2023-09-29/12-07-08_gpu0 \
        --anneal-competition \
        --epochs 1 \
    #   --keep-checkpoints 1 \
    #    --epochs 11 \
done;
