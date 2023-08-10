#!/bin/sh
python main.py \
	--model 'CNN' \
	--task 'MNIST' \
	--channels 32 64 \
	--kernels 5 5 \
	--pools 'mm' \
	--strides 1 1 \
	--fc 10 \
	--optim 'adam' \
	--lrs 5e-5 1e-5 8e-6 \
	--epochs 10 \
	--act 'hard_sigmoid' \
	--todo 'train' \
	--betas 0.0 0.4 \
	--T1 100 \
	--T2 10 \
	--mbs 100 \
	--device 0 \
	--save \
	--seed 710610 \
	--loss cel

