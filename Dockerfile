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
# cf until 1.12.3: https://github.com/tensorflow/tensorflow/blob/v1.12.3/tensorflow/tools/docker/Dockerfile.devel-gpu
# or master https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/dockerfiles/dockerfiles/devel-gpu.Dockerfile
# The difference here is that no jupyter, keras, etc are needed, the goal here is tha bare minimum
# in order to build the tensorflow-gpu with python3.5, cuda 9.1.85 and cudnn 7.1.3

ENV CUDNN_PKG_VERSION 7.1.3.16-1+cuda9.1

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
        libcudnn7=$CUDNN_PKG_VERSION \
        libcudnn7-dev=$CUDNN_PKG_VERSION \
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

# Install bazel, version 0.16.0 can build tensorflow up to version 1.12,
# but for version 1.13 onwards bazel >=0.19.0 is required.
ENV BAZEL_VERSION 0.19.2
WORKDIR /
RUN mkdir /bazel && \
    wget -O /bazel/installer.sh "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh" && \
    wget -O /bazel/LICENSE.txt "https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE" && \
    chmod +x /bazel/installer.sh && \
    /bazel/installer.sh && \
rm -f /bazel/installer.sh

# It is important to build with NCCL from an "agnostic" package, i.e., one that does not come from a specific distro,
# cf: https://github.com/tensorflow/tensorflow/issues/20937#issuecomment-406382691
# and https://github.com/tensorflow/tensorflow/commit/eb54349cb4274ab797917a9e0699decec0b9794c
# I have downloaded it from Nvidia on https://developer.nvidia.com/nccl/nccl-legacy-downloads
# and stored it on a personal Dropbox folder.
RUN mkdir -p /usr/local/nccl2
# Or "https://www.dropbox.com/s/f3r9tkg6wk29pw8/nccl_2.1.15-1+cuda9.1_x86_64.txz" for Cuda 9.2
RUN wget "https://www.dropbox.com/s/f3r9tkg6wk29pw8/nccl_2.1.15-1+cuda9.1_x86_64.txz"
RUN tar -xvf nccl_2.1.15-1+cuda9.1_x86_64.txz --directory /usr/local/nccl2 --strip-components=1

# These packages are required from tensorflow 1.11 onwards
# https://github.com/tensorflow/tensorflow/issues/21518
# https://stackoverflow.com/a/51774943/914874
RUN pip3 install keras_applications==1.0.4 --no-deps
RUN pip3 install keras_preprocessing==1.0.2 --no-deps
RUN pip3 install h5py==2.8.0

# Download and build TensorFlow.
ENV TENSORFLOW_GIT_BRANCH_VERSION r1.14
WORKDIR /tensorflow
RUN git clone --branch=$TENSORFLOW_GIT_BRANCH_VERSION --depth=1 https://github.com/tensorflow/tensorflow.git .

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

# RUN echo "import /tensorflow/tools/bazel.rc" >> /tensorflow/.bazelrc

RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH} \
    tensorflow/tools/ci_build/builds/configured GPU \
    bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --copt=-msse4.1 --copt=-msse4.2 --config=cuda \
    --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" \
        tensorflow/tools/pip_package:build_pip_package && \
    rm /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/pip && \
    rm -rf /root/.cache