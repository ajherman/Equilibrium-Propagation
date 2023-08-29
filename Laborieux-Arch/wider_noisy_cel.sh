#!/bin/sh
python ../main.py \
    --model 'CNN' \
    --task 'CIFAR10' \
    --channels 128 256 512 512 \
    --kernels 5 3 5 3 \
    --pools 'mmmi' \
    --strides 1 1 1 1 \
    --paddings 2 1 1 0 \
    --fc 10 \
    --optim 'sgd' \
    --mmt 0.9 \
    --lrs 0.25 0.15 0.1 0.08 0.05 \
    --lr-decay \
    --wds 3e-4 3e-4 3e-4 3e-4 3e-4 \
    --act 'my_hard_sig' \
    --todo 'train' \
    --betas 0.0 0.5 \
    --thirdphase \
    --T1 250 \
    --T2 25 \
    --mbs 128 \
    --loss 'cel' \
    --softmax \
    --data-aug \
    --save  \
    --device 0 \
    --seed $2 \
    --epochs 120 \
    --noise $1 \
    #--load-path-convert results/EP/cel/2023-07-29/origcode_11-01-25_gpu0_120epochs
# results/EP/mse/2023-08-03/14-38-36_gpu0 \

