#!/bin/bash


export CXX=g++
export CC=gcc

pip install torch
pip install setuptools
pip install ninja
pip install wheel

pip install .


sizes=(512 1024 2048 4096 8192 16384)

for size in "${sizes[@]}"; do
    echo "Running with size $size..."

    ncu --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed \
        --csv python utils/test_flash.py 4 "$size" 32 128 > "${size}.csv"

    echo "Output saved to ${size}.csv"

done


python utils/plot_kernels.py

echo "All done!"

