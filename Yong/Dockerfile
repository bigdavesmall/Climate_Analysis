# Start with CUDA Tensorflow base image
FROM tensorflow/tensorflow:latest-gpu
MAINTAINER Yong Zhuang <jazz14jazz@gmail.com>

# Install dependencies
RUN apt-get update && apt-get install -y \
  libhdf5-dev \
  python-h5py \
  python-yaml

# Upgrade six & install keras
RUN pip install --upgrade six
RUN pip install keras

# Copy sample code.
COPY samples /notebooks/samples
