#!/bin/sh
python ../main.py \
    --load-path $1 \
    --T1 15 \
    --T2 600 \
    --dt 0.01 \
    --eps 0.1 0.2 0.5 0.75 1.0 1.5 2.0 3.0 4.0 6.0 8.0 10.0 20.0 \
    --task CIFAR10 \
    --mbs-test 128 \
    --save \
    --todo attack \
    --nbatches 5 \
    --device $2 \
    --pgdnorm 2 \
    --figs \
    --deltapertubation \
#    --eps 0.001 0.01 0.03 0.07 0.1 0.3 0.5 0.7 1.0 1.5 2.0 2.5 3.0 \
#1000000 \
#    --eps 0.1 0.5 1.0 1.5 2.0 2.5 3.0 4.0 5.0 7.0 10.0 \
