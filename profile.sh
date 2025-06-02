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
#pip install .

echo "Starting profiling"

#ncu -f --target-processes all --set full \
#--import-source on \
#-o profile_flash_attn_causal python utils/test_flash_fwd_causal.py 1 128 1 128 1


ncu -f --target-processes all --set full \
--import-source on \
-o profile_flash_attn_causal python utils/test_flash_backward.py 4 4096 32 128 1
#-o profile_flash_attn python utils/test_flash_fwd_causal.py 4 4096 32 128 0

#-o profile_flash_attn python utils/test_flash_backward.py 4 4096 32 128

#-o profile_flash_attn python utils/test_flash_backward.py 4 4096 32 128

ncu -f --target-processes all --set full \
--import-source on \
-o profile_flash_attn python utils/test_flash_backward.py 4 4096 32 128 0

echo "All done!"
