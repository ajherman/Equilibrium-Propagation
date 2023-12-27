#!/bin/sh
python ./main.py \
    --load-path Laborieux-Arch/results/EP/mse/2023-08-12/08-24-23_gpu0 \
    --T1 15 \
    --T2 250 \
    --task CIFAR10 \
    --mbs-test 256 \
    --save \
    --eps 0.05 0.1 0.25 0.4 0.5 0.6 0.75 1.0 1.25 1.5 1.75 \
    --todo attack \
    --nbatches 5
