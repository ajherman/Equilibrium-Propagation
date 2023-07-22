#!/bin/sh
python ../main.py \
	--model 'CNN' \
	--task 'CIFAR10' \
	--channels 128 256 512 512 \
	--kernels 3 3 3 3 \
	--pools 'mmmm' \
	--strides 1 1 1 1 \
	--paddings 1 1 1 0 \
	--fc 10 \
	--softmax \
	--optim 'sgd' \
	--mmt 0.9 \
	--lrs 0.25 0.15 0.1 0.08 0.05 \
	--lr-decay \
	--epochs 120 \
	--wds 3e-4 3e-4 3e-4 3e-4 3e-4 \
	--act 'mysig' \
	--todo 'train' \
	--betas 0.0 0.1 \
	--thirdphase \
	--T1 150 \
	--T2 25 \
	--mbs 128 \
	--loss 'cel' \
	--save  \
	--device 0 \
	--load-path results/EP/cel/2023-07-18/12-05-15_gpu0

