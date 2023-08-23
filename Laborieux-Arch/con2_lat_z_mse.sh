#!/usr/bin/sh
python ../main.py --model LateralCNN --task CIFAR10 --channels 128 256 512 512 --kernels 3 3 3 3 --pools mmmm --strides 1 1 1 1 --paddings 1 1 1 0 --fc 10 --optim sgd --mmt 0.9 --lrs 0.25 0.15 0.1 0.08 0.05 --lat-layers -3 -2 -1 --lat-lrs 0.05 0.1 0.15 --train-lateral --lat-constraints zerodiag,transposesymmetric zerodiag,transposesymmetric zerodiag,transposesymmetric --lat-init-zeros --lr-decay --epochs 120 --wds 3e-4 3e-4 3e-4 3e-4 3e-4 --lat-wds 3e-4 3e-4 3e-4 --act my_hard_sig --todo train --betas 0.0 0.5 --thirdphase --T1 250 --T2 25 --mbs 128 --data-aug --loss mse --save --device 0 --seed 292 --load-path results/EP/mse/2023-08-15/16-45-01_gpu0
