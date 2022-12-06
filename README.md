# DataMinig Team 5

## simple test for evaluation
`./src/exe.sh`

## evaluation with parameter setting
- DATA: data location (deprecated)
- TOP_K: recommendation items 
- COMB_L: left combination parameter
- COMB_R: right combiation parameter
- W: alpha of weighted sum in pre-processing
- DATASIZE: data size (currently we use 100k data)  


    # arguments
    DATA="../data/ml-25m"
    TOP_K="10"
    DATASIZE="100k" # Select MovieLens data size: 100k, 1m, 10m, or 20m
    COMB_L="20"
    COMB_R="3"
    W="0.9"

    # for debugging
    # ARGS="--data-path $DATA --top-k $TOP_K --comb $COMB --data-size $DATASIZE --comb-l $COMB_L --comb-r $COMB_R"
    # python3 main.py $ARGS

    # for evaluation
    for i in $DATASIZE; do
        echo "data size $i evaluation..."
        ARGS="--data-path $DATA --top-k $TOP_K --comb $COMB --data-size $i --comb-l $COMB_L --comb-r $COMB_R --weight $W"
        python3 main.py $ARGS
    done
