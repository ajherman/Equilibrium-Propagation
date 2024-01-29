#!/bin/sh
python ../main.py \
    --load-path $1 \
    --T1 $3 \
    --T2 $3 \
    --dt 0.01 \
    --eps 0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.3 1.0 \
    --task CIFAR10 \
    --mbs-test 128 \
    --save \
    --todo attack_blackbox \
    --nbatches 1 \
    --device $2 \
    --norm 1000000000 \
#    --eps 0.001 0.01 0.03 0.07 0.1 0.3 0.5 0.7 1.0 1.5 2.0 2.5 3.0 \
#1000000 \
#    --eps 0.1 0.5 1.0 1.5 2.0 2.5 3.0 4.0 5.0 7.0 10.0 \
