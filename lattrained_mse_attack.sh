#!/bin/sh
python ./main.py --model LateralCNN  \
    --todo attack --device 0 --seed 303 \
    --load-path Laborieux-Arch/results/EP/mse/2023-08-15/17-45-32_gpu0 \
    --loss mse \
    --T1 15 \
    --T2 250 \
    --task CIFAR10 \
    --mbs-test 256 \
    --save \
    --todo attack \
    --nbatches 5 \
    --eps 0.1 0.25 0.4 0.5 0.6 0.75 1.0 1.25 1.5 1.75 \
