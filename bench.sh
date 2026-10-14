#!/bin/bash
SIZES="128 256 384 512 768 1024 1280 1536 2048"

echo "N,time_ms" > cpu_results.csv
echo "N,kernel_ms,h2d_ms,d2h_ms,total_ms" > gpu_results.csv

for N in $SIZES; do
    for run in 1 2 3; do
        echo "Running N=$N (run $run)"

        # CPU prints "... N=512): 0.137 seconds" -> take the number, convert to ms
        ms=$(./matrix_cpu $N | awk '{print $(NF-1) * 1000}')
        echo "$N,$ms" >> cpu_results.csv

        # GPU prints "RESULT,N,kernel,h2d,d2h,total" -> keep everything after RESULT,
        ./matrix_gpu $N | grep RESULT | cut -d, -f2- >> gpu_results.csv
    done
done

echo "Done: cpu_results.csv and gpu_results.csv"
