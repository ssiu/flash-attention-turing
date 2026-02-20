#!/bin/bash


export CXX=g++
export CC=gcc

# set ninja workers to 2, otherwise leads to OOM on colab
export MAX_JOBS=2

pip install torch
pip install setuptools
pip install ninja
pip install wheel

pip install -v .