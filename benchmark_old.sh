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

batch_size=4
num_heads=16
num_heads_k=16
hdims=(64 128)
seqlens=(512 1024 2048 4096 8192 16384 500 1000 2000 4000 8000 16000)
# seqlens=(8192 16384 8000 16000)
is_causals=(False True)
# sizes=(500 1000 2000 4000 8000 16000)

KERNEL_REGEX='(flash_.*|fmha_cutlass.*)'

for seqlen in "${seqlens[@]}"; do
    for hdim in "${hdims[@]}"; do
        for is_causal in "${is_causals[@]}"; do
            echo "Running with size $seqlen, $hdim, $is_causal..."

            ncu --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed \
                --kernel-name-base demangled \
                --kernel-name ::regex:"${KERNEL_REGEX}" \
                --csv python -c "import torch; from test_flash_attn_old import test_flash_attn_bwd; test_flash_attn_bwd(${batch_size}, ${num_heads}, ${num_heads_k}, ${seqlen}, ${seqlen}, ${hdim}, ${is_causal}, torch.float16)" \
                > "${seqlen}_${hdim}_${is_causal}.csv"

            echo "Finished ${seqlen}_${hdim}_${is_causal}.csv"
        done
    done
done


python utils/plot_kernels.py

echo "All done!"
