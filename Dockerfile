FROM ubuntu:16.04

# == tensorflow-gpu: tensorflow + cuda and cudnn build with bazel based on the official Dockerfile
# cf until 1.12.3: https://github.com/tensorflow/tensorflow/blob/v1.12.3/tensorflow/tools/docker/Dockerfile.devel-gpu
# or master https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/dockerfiles/dockerfiles/devel-gpu.Dockerfile
# The difference here is that no jupyter, keras, etc are needed, the goal here is tha bare minimum
# in order to build the tensorflow-gpu with python3.5, cuda 9.1.85 and cudnn 7.1.3

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
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
    rm -rf /var/lib/apt/lists/*

# Set up Bazel.

# Running bazel inside a `docker build` command causes trouble, cf:
#   https://github.com/bazelbuild/bazel/issues/134
# The easiest solution is to set up a bazelrc file forcing --batch.
RUN echo "startup --batch" >>/etc/bazel.bazelrc
# Similarly, we need to workaround sandboxing issues:
#   https://github.com/bazelbuild/bazel/issues/418
RUN echo "build --spawn_strategy=standalone --genrule_strategy=standalone" >>/etc/bazel.bazelrc

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

# These packages are required from tensorflow 1.11 onwards
# https://github.com/tensorflow/tensorflow/issues/21518
# https://stackoverflow.com/a/51774943/914874
RUN pip3 install keras_applications==1.0.4 --no-deps
RUN pip3 install keras_preprocessing==1.0.2 --no-deps
RUN pip3 install h5py==2.8.0

# Download and build TensorFlow.
ENV TENSORFLOW_GIT_BRANCH_VERSION r1.13
WORKDIR /tensorflow
RUN git clone --branch=$TENSORFLOW_GIT_BRANCH_VERSION --depth=1 https://github.com/tensorflow/tensorflow.git .

ENV TF_NEED_GCP=0 \
    TF_NEED_HDFS=0 \
    TF_NEED_OPENCL=0 \
    TF_NEED_JEMALLOC=0 \
    TF_ENABLE_XLA=0 \
    TF_NEED_VERBS=0 \
    TF_CUDA_CLANG=0 \
    TF_DOWNLOAD_CLANG=0 \
    TF_NEED_MKL=1 \
    TF_DOWNLOAD_MKL=1 \
    TF_NEED_MPI=0 \
    TF_NEED_S3=0 \
    TF_NEED_KAFKA=0 \
    TF_NEED_GDR=0 \
    TF_NEED_OPENCL_SYCL=0 \
    TF_SET_ANDROID_WORKSPACE=0 \
    TF_NEED_AWS=0 \
    CI_BUILD_PYTHON=python3 \
    TF_NEED_CUDA=0 \
    TF_NEED_TENSORRT=0 \
    TMP="/tmp"

# To workaround an issue on python3-only environment, cf: https://github.com/tensorflow/tensorflow/pull/19443 and
# https://github.com/tensorflow/tensorflow/issues/15618
RUN ln -s -f /usr/bin/python3 /usr/bin/python

RUN tensorflow/tools/ci_build/builds/configured CPU \
    bazel build --config=mkl -c opt \
        --copt=-mavx \
        --copt=-mavx2 \
        --copt=-mfma \
        --copt=-mfpmath=both \
        --copt=-msse4.1 \
        --copt=-msse4.2 \
        # For optimized builds appropriate for the hardware platform of your choosing, uncomment below...
        # For ivy-bridge or sandy-bridge
        # --copt=-march="ivybridge" \
        # for haswell, broadwell, or skylake
        --copt=-march="haswell" \
        --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" \
        tensorflow/tools/pip_package:build_pip_package && \
        bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/pip && \
        rm -rf /root/.cache