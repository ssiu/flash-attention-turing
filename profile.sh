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


batch_size=4
num_heads=16
num_heads_k=16
hdims=(128)
seqlens=(8192)
# seqlens=(8192 16384 8000 16000)
is_causals=(False True)
softmax_scale=None
# sizes=(500 1000 2000 4000 8000 16000)
KERNEL_REGEX='flash_fwd_.*'

for seqlen in "${seqlens[@]}"; do
    for hdim in "${hdims[@]}"; do
        for is_causal in "${is_causals[@]}"; do
            echo "Running with size $seqlen, $hdim, $is_causal..."

            ncu -f --target-processes all --set full \
                --import-source on \
                --kernel-name-base demangled \
                --kernel-name ::regex:"${KERNEL_REGEX}" \
                -o "${seqlen}_${hdim}_${is_causal}" \
                python -c "import torch; from test_flash_attn import test_flash_attn; test_flash_attn(${batch_size}, ${num_heads}, ${num_heads_k}, ${seqlen}, ${seqlen}, ${hdim}, ${softmax_scale}, ${is_causal}, torch.float16)"

            echo "Finished ${seqlen}_${hdim}_${is_causal}"
        done
    done
done




# ncu -f --target-processes all --set full \
# --import-source on \
# -o profile_flash_attn_64_causal python utils/test_flash_fwd_causal.py 4 4096 32 64 1
#
#ncu -f --target-processes all --set full \
#--import-source on \
#-o profile_flash_attn_128 python utils/test_flash_fwd_causal.py 4 4096 32 128 0
#
#ncu -f --target-processes all --set full \
#--import-source on \
#-o profile_flash_attn_128_causal python utils/test_flash_fwd_causal.py 4 4096 32 128 1


# ncu -f --target-processes all --set full \
# --import-source on \
# -o profile_flash_attn_64 python utils/test_flash_backward.py 4 4096 32 64 0

# ncu -f --target-processes all --set full \
# --import-source on \
# -o profile_flash_attn_64_causal python utils/test_flash_backward.py 4 4096 32 64 1

# ncu -f --target-processes all --set full \
# --import-source on \
# -o profile_flash_attn_128 python utils/test_flash_backward.py 4 4096 32 128 0

# ncu -f --target-processes all --set full \
# --import-source on \
# -o profile_flash_attn_128_causal python utils/test_flash_backward.py 4 4096 32 128 1








#-o profile_flash_attn python utils/test_flash_backward.py 4 4096 32 128

#-o profile_flash_attn python utils/test_flash_backward.py 4 4096 32 128

#ncu -f --target-processes all --set full \
#--import-source on \
#-o profile_flash_attn python utils/test_flash_backward.py 4 4096 32 128 0

echo "All done!"
