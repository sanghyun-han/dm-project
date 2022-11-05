#!/bin/bash

# arguments
DATA="../data/ml-25m"
TOP_K="10"
COMB="3"
DATASIZE="100k" # Select MovieLens data size: 100k, 1m, 10m, or 20m

ARGS="--data-path $DATA --top-k $TOP_K --comb $COMB --data-size $DATASIZE"
python3 main.py $ARGS