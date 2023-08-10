#!/bin/sh
python ./main.py \
    --load-path Laborieux-Arch/results/EP/cel/2023-07-29/origcode_11-01-25_gpu0_120epochs \
    --T1 15 \
    --T2 250 \
    --task CIFAR10 \
    --mbs-test 256 \
    --save \
    --todo attack \
    --nbatches 5 \
    --eps 0.1 0.25 0.4 0.5 0.6 0.75 1.0 1.25 1.5 1.75 \
