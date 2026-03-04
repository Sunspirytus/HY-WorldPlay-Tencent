#!/bin/bash

cp /root/.bashrc /app/.bashrc

source /app/.bashrc

echo "\
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud" \
>> ${MAMBA_ROOT_PREFIX}/.condarc

micromamba install --yes --name base python=3.11
micromamba run --name base pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/simple

cd /app

micromamba run --name base pip install -r requirements.txt

export HTTPS_PROXY=http://172.18.4.17:10888

micromamba run --name base pip install flash-attn --no-build-isolation
micromamba run --name base pip install angelslim==0.2.2 gradio==6.8.0

git clone --recursive -b v2.1.1.post3 git@github.com:deepseek-ai/DeepGEMM.git
cd DeepGEMM
./develop.sh
./install.sh

unset HTTPS_PROXY
