#!/bin/sh
python $(head -1 $1/hyperparameters.txt) --load-path $1 --epochs $2
