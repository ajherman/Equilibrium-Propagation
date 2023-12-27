#!/bin/sh

for seed in 111 222; do
    for dt in 1.0 0.5 0.1 0.01; do
        T1=$(perl -e "print 250/$dt");
        T2=$(perl -e "print 25/$dt");
        bash ./sparsecodingcnn.sh $seed $dt $T1 $T2
    done;
done;
