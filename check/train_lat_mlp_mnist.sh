#!/bin/sh
python main.py --model 'LatMLP' --task 'MNIST' --archi 784 512 10 --optim 'adam' --lrs 5e-4 5e-4 --lat-lrs 5e-4 5e-4 --epochs 20 --act 'mysig' --todo 'train' --betas 0.0 0.1 --T1 30 --T2 10 --mbs 50 --device 0 --save --use-lateral
