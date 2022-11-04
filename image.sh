#!/bin/bash

# docker pull nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04
# docker cp MLNX_OFED_LINUX-5.4-3.4.0.0-ubuntu20.04-x86_64.iso cuda113:/tmp

# check cuda image download
tmp=$(docker images)
if echo $tmp | grep -q "cudnn8-devel"; then
    echo "find cuda image... skip download"
else
    echo "cannot find cuda image... download"
    docker pull nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04
fi

# check cuda image name exist
tmp=$(docker ps -a)
if docker run -it -d --privileged --name cuda113 nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04 2> /dev/null; then
    echo "make cuda image... cuda113"
fi

docker commit cuda113 dm_cuda113

# make docker image with pytorch
docker build --tag dm-project .