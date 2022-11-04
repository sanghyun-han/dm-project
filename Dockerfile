FROM dm_cuda113

LABEL maintainer="Sanghyun Han sanghyun@snu.ac.kr"

ENV DEBIAN_FRONTEND noninteractive

ARG PYTHON_VERSION=3.8
ARG PIP_VERSION=3

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        cmake \
        curl \
        git \
        libjpeg-dev \
        net-tools \
        iputils-ping \
        libssl-dev \
        libpng-dev && \
        rm -rf /var/lib/apt/lists/*

# pytorch dependency
RUN apt update
RUN apt-get install -y python${PYTHON_VERSION}
RUN apt install -y python${PIP_VERSION}-pip
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

RUN mkdir /workspace

COPY ./ /workspace/