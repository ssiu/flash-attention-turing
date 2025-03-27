#!/bin/bash


export CXX=g++
export CC=gcc

pip install torch
pip install setuptools
pip install ninja
pip install wheel

pip install .


ncu -f --target-processes all --set full \
--import-source on \
-o profile_flash_attn python utils/test_flash.py 4 4096 32 128


echo "All done!"