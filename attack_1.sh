#!/bin/sh
python ./main.py --load-path Laborieux-Arch/results/EP/cel/2023-07-24/09-30-41_gpu0_84epochs/ --T1 15 --T2 250 --task CIFAR10 --mbs-test 256 --data-aug --save --eps 1.0 2.0 3.0 4.0 5.0 --todo attack --figs --nbatches 2
