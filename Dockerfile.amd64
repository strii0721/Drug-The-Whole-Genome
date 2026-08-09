# syntax=docker/dockerfile:1.4

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04
ARG http_proxy
ARG https_proxy
ARG python=3.9
# Set default RUN shell to /bin/bash
SHELL ["/bin/bash", "-cu"]


# Set environment variables
ENV LANG=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHON_VERSION=${python} \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"

# 配置国内 apt 源
RUN sed -i 's@//.*archive.ubuntu.com@//mirrors.aliyun.com@g' /etc/apt/sources.list \
    && sed -i 's@//.*security.ubuntu.com@//mirrors.aliyun.com@g' /etc/apt/sources.list
# Install basic packages for compiling and building
RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
    build-essential \
    cmake \
    g++-7 \
    git \
    curl \
    wget \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    librdmacm1 \
    libibverbs1 \
    ibverbs-providers \
    tzdata \
    libgl1-mesa-glx \
    libglib2.0-0 \
    fontconfig \
    && rm -rf /var/lib/apt/lists/*


# Install Miniconda (pinned to a Python 3.9 build).
# NOTE: "Miniconda3-latest" no longer resolves `conda install python=3.9`
# (its meta-packages require newer Python), so we pin a py39 installer.
# NOTE: `conda tos accept` was removed in modern conda (24.x); the pinned
# py39 installer does not ship the `conda-tos` plugin, so ToS is never
# prompted and these lines must be omitted.
ARG MINICONDA=Miniconda3-py39_24.5.0-0-Linux-x86_64.sh
RUN wget https://repo.anaconda.com/miniconda/${MINICONDA} -O /tmp/install-conda.sh \
    && chmod +x /tmp/install-conda.sh \
    && bash /tmp/install-conda.sh -b -f -p /usr/local \
    && rm -f /tmp/install-conda.sh \
    && conda clean -y --all

# ==================================================
# *Tools installation (optional)
#
# You find some tools useful or unwanted, comment or uncomment them as you wish.
# Check this page for more installation commands:
# https://github.com/hughplay/memo/blob/master/scripts/prepare_dl.sh
# ==================================================


# Install basic packages
RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
    zsh \
    vim \
    zip \
    unzip \
    rsync \
    htop \
    language-pack-en \
    nethogs \
    sysstat \
    gnupg \
    lsb-release \
    sudo \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Install lightvim, a configuration for vim
# RUN wget https://raw.githubusercontent.com/hughplay/lightvim/master/install.sh -O - | sh


# ==================================================
# *Fonts installation
#
# Install fonts that are commonly used in plotting.
# Question: "Which font looks best in a scientific figure?"
# Answer: "Arial or Helvetica, always."
# https://pubs.acs.org/doi/10.1021/acs.chemmater.6b00306
# ==================================================

RUN cd /tmp \
    && wget https://github.com/hughplay/memo/raw/master/code/snippet/drawing/plot_fonts.tar.gz \
    && tar zxvf plot_fonts.tar.gz \
    && cp plot_fonts/* /usr/share/fonts/truetype/ \
    && rm -rf plot_fonts.tar.gz plot_fonts \
    && fc-cache -fv

# ==================================================
# Install python packages
# ==================================================

# Setup TUNA mirror (optional)
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN cat <<EOT >> ~/.condarc
channels:
  - defaults
show_channel_urls: true
channel_alias: https://mirrors.tuna.tsinghua.edu.cn/anaconda
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/pro
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  nvidia: https://mirrors.sustech.edu.cn/anaconda-extra/cloud
EOT




# Install Pytorch 2.1.2
# you can find other versions and installation commands from:
# https://pytorch.org/get-started/previous-versions/
# https://github.com/pytorch/pytorch/wiki/PyTorch-Versions
# NOTE: `conda tos accept` removed in conda 24.x; omitted.
RUN conda install -y \
    pytorch==2.1.2 \
    torchvision==0.16.2 \
    torchaudio==2.1.2 \
    pytorch-cuda==12.1 \
    faiss-gpu=1.8.0 \
    -c pytorch -c nvidia -c conda-forge \
    && conda clean -afy
# Install Uni-core
# NOTE: Uni-Core's setup.py pulls `wandb` -> `pydantic` (2.x), which requires
# typing-extensions>=4.15.0. The conda pytorch install drops typing-extensions
# 4.14.1, so pip fails to resolve during `python setup.py install` (exit 1).
# Bump it here first; 4.15.x is backward-compatible with torch 2.1.2 (needs >=4.8.0).
RUN pip install --no-cache-dir \
    iopath \
    lmdb \
    ml_collections \
    numpy \
    scipy \
    tensorboardX \
    tqdm \
    tokenizers \
    "typing-extensions>=4.15.0"

# NOTE: `python setup.py install` resolves install_requires via the legacy
# easy_install, which cannot build modern PEP 517 sdists (pydantic-core 2.47.0
# ships a pyproject-based sdist with no top-level setup.py -> "Couldn't find a
# setup script"). Use pip instead. `--no-build-isolation` is REQUIRED because
# Uni-Core's setup.py does `import torch` at module top, so the build needs the
# already-installed torch from the current env (not an isolated build env).
RUN git clone https://github.com/dptech-corp/Uni-Core.git /tmp/Uni-core \
    && cd /tmp/Uni-core \
    && pip install --no-build-isolation . \
    && rm -rf /tmp/Uni-core


# By default, install packages from `requirements.txt` with pip.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm -f /tmp/requirements.txt

# Another way is installing packages from a `env.yaml` with conda.
# COPY env.yaml /tmp/env.yaml
# RUN conda env create -f /tmp/env.yaml && rm -f /tmp/env.yaml


# ==================================================
# Post-installation steps
#
# Create a user that has the same UID and GID as the host user. This will
# prevent many privileges issues.
# ==================================================


# TUNA mirror for apt
# RUN sed -i -e 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list

# Add a user with the same UID and GID as the host user, to prevent privilege issues.
# ARG USER_ID=1011
# ARG GROUP_ID=1011
# ARG USER_NAME=docker
# RUN if [ $USER_NAME != "root" ] ; \
#     then addgroup --gid ${GROUP_ID} ${USER_NAME} \
#     && adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID ${USER_NAME} \
#     && usermod -aG sudo ${USER_NAME} \
#     && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers ; fi


# # Copy installed configuration files from root to user
# # COPY misc/init_workspace /usr/local/bin
# RUN chmod +x /usr/local/bin/init_workspace
# RUN /usr/local/bin/init_workspace --user ${USER_NAME} --home /home/${USER_NAME}


# # backup $HOME for reverse mounting $HOME
# RUN rsync -a /home/${USER_NAME}/ /${USER_NAME}_home_bak


# # Switch to the created user
# USER ${USER_NAME}


# Set working directory to /project
WORKDIR "/project"
