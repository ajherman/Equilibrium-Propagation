#!/bin/sh

for i in {0..10}; do
    echo '';
    echo "====== ATTACKING MODEL STEP $i =======";
    echo '';

    ./attack-long.sh $1/model_$i.pt $2; 
done;
