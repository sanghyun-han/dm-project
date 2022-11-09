#!/bin/bash

# arguments
DATA="../data/ml-25m"
TOP_K="10"
COMB="3"
DATASIZE="100k 1m 10m 20m" # Select MovieLens data size: 100k, 1m, 10m, or 20m
COMB_L="20"
COMB_R="3"

# for debugging
# ARGS="--data-path $DATA --top-k $TOP_K --comb $COMB --data-size $DATASIZE --comb-l $COMB_L --comb-r $COMB_R"
# python3 main.py $ARGS

# for evaluation
for i in $DATASIZE; do
    echo "data size $i evaluation..."
    ARGS="--data-path $DATA --top-k $TOP_K --comb $COMB --data-size $i --comb-l $COMB_L --comb-r $COMB_R"
    python3 main.py $ARGS
done