# Run:
#    sudo docker build -t build-tensorflow .
# Then copy the tensorflow wheel:
#    sudo docker cp CONTAINER_ID:/tmp/pip/tensorflow-1.7.1-cp35-cp35m-linux_x86_64.whl .
# To get the container id: docker ps -alq or docker ps then see the CONTAINER_ID respective
# to the "build-tensorflow" image.

# == nvidia-base: a raw ubuntu:16.04 with specific nvidia repositories configured
# cf: https://gitlab.com/nvidia/cuda/blob/ubuntu16.04/9.1/base/Dockerfile

FROM ubuntu:16.04

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates apt-transport-https gnupg-curl && \
    rm -rf /var/lib/apt/lists/* && \
    NVIDIA_GPGKEY_SUM=d1be581509378368edeec8c1eb2958702feedf3bc3d17011adbf24efacce4ab5 && \
    NVIDIA_GPGKEY_FPR=ae09fe4bbd223a84b2ccfce3f60f4b3d7fa2af80 && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub && \
    apt-key adv --export --no-emit-version -a $NVIDIA_GPGKEY_FPR | tail -n +5 > cudasign.pub && \
    echo "$NVIDIA_GPGKEY_SUM  cudasign.pub" | sha256sum -c --strict - && rm cudasign.pub && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

ENV CUDA_VERSION 9.1.85

ENV CUDA_PKG_VERSION 9-1=$CUDA_VERSION-1
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-cudart-$CUDA_PKG_VERSION && \
    ln -s cuda-9.1 /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*

# nvidia-docker 1.0
LABEL com.nvidia.volumes.needed="nvidia_driver"
LABEL com.nvidia.cuda.version="${CUDA_VERSION}"

RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=9.1"

# == tensorflow-gpu: tensorflow + cuda and cudnn build with bazel based on the official Dockerfile
# cf: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/docker/Dockerfile.devel-gpu
# The difference here is that no jupyter, keras, etc are needed, the goal here is tha bare minimum
# in order to build the tensorflow-gpu with python3.5, cuda 9.1.85 and cudnn 7.1.3

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-9-1 \
        cuda-cublas-dev-9-1 \
        cuda-cudart-dev-9-1 \
        cuda-cufft-dev-9-1 \
        cuda-curand-dev-9-1 \
        cuda-cusolver-dev-9-1 \
        cuda-cusparse-dev-9-1 \
        curl \
        git \
        libcudnn7=7.1.3.16-1+cuda9.1 \
        libcudnn7-dev=7.1.3.16-1+cuda9.1 \
        libcurl3-dev \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python3-dev \
        python3-pip \
        python3-numpy \
        python3-setuptools \
        python3-scipy \
        python3-wheel \
        rsync \
        software-properties-common \
        unzip \
        zip \
        zlib1g-dev \
        wget \
        && \
    rm -rf /var/lib/apt/lists/* && \
    find /usr/local/cuda-9.1/lib64/ -type f -name 'lib*_static.a' -not -name 'libcudart_static.a' -delete && \
    rm /usr/lib/x86_64-linux-gnu/libcudnn_static_v7.a

#RUN apt-get update && \
#        apt-get install nvinfer-runtime-trt-repo-ubuntu1604-4.0.1-ga-cuda9.0 && \
#        apt-get update && \
#        apt-get install libnvinfer4=4.1.2-1+cuda9.0 && \
#        apt-get install libnvinfer-dev=4.1.2-1+cuda9.0

# Set up Bazel.

# Running bazel inside a `docker build` command causes trouble, cf:
#   https://github.com/bazelbuild/bazel/issues/134
# The easiest solution is to set up a bazelrc file forcing --batch.
RUN echo "startup --batch" >>/etc/bazel.bazelrc
# Similarly, we need to workaround sandboxing issues:
#   https://github.com/bazelbuild/bazel/issues/418
RUN echo "build --spawn_strategy=standalone --genrule_strategy=standalone" >>/etc/bazel.bazelrc
# Install bazel
ENV BAZEL_VERSION 0.16.0
WORKDIR /
RUN mkdir /bazel && \
    cd /bazel && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -o /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE && \
    chmod +x bazel-*.sh && \
    ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    cd / && \
    rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh

# It is important to build with NCCL from an "agnostic" package, i.e., one that does not come from a specific distro,
# cf: https://github.com/tensorflow/tensorflow/issues/20937#issuecomment-406382691
# and https://github.com/tensorflow/tensorflow/commit/eb54349cb4274ab797917a9e0699decec0b9794c
# I have downloaded it from Nvidia on https://developer.nvidia.com/nccl/nccl-legacy-downloads
# and stored it on a personal Dropbox folder.
RUN mkdir -p /usr/local/nccl2
RUN wget "https://www.dropbox.com/s/f3r9tkg6wk29pw8/nccl_2.1.15-1+cuda9.1_x86_64.txz"
RUN tar -xvf nccl_2.1.15-1+cuda9.1_x86_64.txz --directory /usr/local/nccl2 --strip-components=1

# Download and build TensorFlow.
WORKDIR /tensorflow
RUN git clone --branch=r1.7 --depth=1 https://github.com/tensorflow/tensorflow.git .

# Configure the build for our CUDA configuration.
ENV TF_NCCL_VERSION='2.1.15'
ENV NCCL_INSTALL_PATH=/usr/local/nccl2
ENV TF_NEED_GCP 0
ENV TF_NEED_HDFS 0
ENV TF_NEED_OPENCL 0
ENV TF_NEED_JEMALLOC 0
ENV TF_ENABLE_XLA 0
ENV TF_NEED_VERBS 0
ENV TF_CUDA_CLANG 0
ENV TF_DOWNLOAD_CLANG 0
ENV TF_NEED_MKL 0
ENV TF_DOWNLOAD_MKL 0
ENV TF_NEED_MPI 0
ENV TF_NEED_S3 0
ENV TF_NEED_KAFKA 0
ENV TF_NEED_GDR 0
ENV TF_NEED_OPENCL_SYCL 0
ENV TF_SET_ANDROID_WORKSPACE 0
ENV TF_NEED_AWS 0
ENV CI_BUILD_PYTHON python3
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV TF_NEED_CUDA 1
ENV TF_NEED_TENSORRT 0
ENV TF_CUDA_COMPUTE_CAPABILITIES=3.5,5.2,6.0,6.1,7.0
ENV TF_CUDA_VERSION=9.1
ENV TF_CUDNN_VERSION=7
ENV TMP="/tmp"

# To workaround an issue on python3-only environment, cf: https://github.com/tensorflow/tensorflow/pull/19443 and
# https://github.com/tensorflow/tensorflow/issues/15618
RUN ln -s -f /usr/bin/python3 /usr/bin/python

RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH} \
    tensorflow/tools/ci_build/builds/configured GPU \
    bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --copt=-msse4.1 --copt=-msse4.2 --config=cuda \
    --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" \
        tensorflow/tools/pip_package:build_pip_package && \
    rm /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/pip && \
    rm -rf /root/.cache

#FROM debian:stretch
#
#RUN echo 'deb http://deb.debian.org/debian stretch contrib non-free' >> /etc/apt/sources.list
#RUN echo 'deb http://deb.debian.org/debian stretch-backports main contrib non-free' >> /etc/apt/sources.list
#
#RUN apt-get update && apt-get install -y \
#        wget \
#        nano \
#        build-essential \
#        curl \
#        git \
#        unzip \
#        zip \
#        libcurl3-dev \
#        pkg-config \
#        python-dev \
#        software-properties-common \
#        python3 \
#        python3-dev \
#        python3-pip
#
#RUN wget "https://www.dropbox.com/s/1kkj8b98ivbrv5a/cuda-repo-ubuntu1604-9-1-local_9.1.85-1_amd64.deb"
#RUN wget "https://www.dropbox.com/s/s5zp0we82ta304j/libcudnn7-dev_7.0.5.15-1+cuda9.1_amd64.deb"
#RUN wget "https://www.dropbox.com/s/abu6fhbcpww9k68/libcudnn7_7.0.5.15-1+Bcuda9.1_amd64.deb"
#RUN dpkg -i *.deb
#RUN apt-key add /var/cuda-repo-9-1-local/7fa2af80.pub
#RUN apt-get install -f

#RUN apt-get install -t stretch-backports -y \
#        nvidia-cuda-dev \
#        nvidia-cuda-toolkit \
#        && \
#    rm -rf /var/lib/apt/lists/*
#
#
## Bazel
#
## Running bazel inside a `docker build` command causes trouble, cf:
##   https://github.com/bazelbuild/bazel/issues/134
## The easiest solution is to set up a bazelrc file forcing --batch.
#RUN echo "startup --batch" >>/etc/bazel.bazelrc
## Similarly, we need to workaround sandboxing issues:
##   https://github.com/bazelbuild/bazel/issues/418
#RUN echo "build --spawn_strategy=standalone --genrule_strategy=standalone" \
#    >>/etc/bazel.bazelrc
#
#ENV BAZEL_VERSION 0.16.0
#WORKDIR /
#RUN mkdir /bazel && \
#    cd /bazel && \
#    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
#    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -o /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE && \
#    chmod +x bazel-*.sh && \
#    ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
#    cd / && \
#    rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh
#
## Tensorflow
#WORKDIR /tensorflow
#RUN git clone --branch=r1.8 --depth=1 https://github.com/tensorflow/tensorflow.git .
#
## Configure the build for our CUDA configuration.
#
#ENV TF_NEED_GCP=0 \
#    TF_NEED_HDFS=0 \
#    TF_NEED_OPENCL=0 \
#    TF_NEED_JEMALLOC=0 \
#    TF_ENABLE_XLA=0 \
#    TF_NEED_VERBS=0 \
#    TF_CUDA_CLANG=0 \
#    TF_DOWNLOAD_CLANG=0 \
#    TF_NEED_MKL=0 \
#    TF_DOWNLOAD_MKL=0 \
#    TF_NEED_MPI=0 \
#    TF_NEED_S3=0 \
#    TF_NEED_KAFKA=0 \
#    TF_NEED_GDR=0 \
#    TF_NEED_OPENCL_SYCL=0 \
#    TF_SET_ANDROID_WORKSPACE=0 \
#    TF_NEED_AWS=0 \
#    CI_BUILD_PYTHON=python3 \
#    LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
#    TF_NEED_TENSORRT=1 \
#    TF_NEED_CUDA=1 \
#    TF_CUDA_COMPUTE_CAPABILITIES=3.5,5.2,6.0,6.1,7.0 \
#    TF_CUDA_VERSION=9.1 \
#    TF_CUDNN_VERSION=7 \
#    CUDA_TOOLKIT_PATH=/usr/lib/x86_64-linux-gnu \
#    CUDNN_INSTALL_PATH=/usr/lib/x86_64-linux-gnu
#
#RUN mkdir -p /usr/local/cuda/
#RUN ln -s /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64
#RUN ln -s /usr/local/cuda/ /usr/lib/x86_64-linux-gnu/lib64
#
#RUN LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH} \
#    tensorflow/tools/ci_build/builds/configured GPU \
#    bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --copt=-msse4.1 --copt=-msse4.2 --config=cuda \
#   --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" \
#        tensorflow/tools/pip_package:build_pip_package && \
#    bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/pip


#RUN ln -s /usr/lib/x86_64-linux-gnu/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1 && \
#    LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:${LD_LIBRARY_PATH} \
#    tensorflow/tools/ci_build/builds/configured GPU \
#    bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --copt=-msse4.1 --copt=-msse4.2 --config=cuda \
#   --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" \
#        tensorflow/tools/pip_package:build_pip_package && \
#    rm /usr/lib/x86_64-linux-gnu/stubs/libcuda.so.1 && \
#    bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/pip

