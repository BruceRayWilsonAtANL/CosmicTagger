#!/bin/bash
# git checkout
# -x ENABLE_CONSOLE=true -x LOG_LEVEL_ALL=3
for i in {1,}
do
    export name=bfloat16_2x10_${i}
    mpirun -v \
        --allow-run-as-root \
        --mca \
        --bind-to core \
        --rank-by core \
        --report-bindings \
        --map-by ppr:4:socket:PE=7 \
        -np 16 \
        --mca btl_tcp_if_include 192.168.201.0/24 \
        --merge-stderr-to-stdout \
        --prefix $MPI_ROOT \
        -H 140.221.77.101:8,140.221.77.102:8 \
        -x GC_KERNEL_PATH \
        -x MASTER_ADDR \
        -x MASTER_PORT \
        -x HCCL_SOCKET_IFNAME=enp75s0f0 \
        -x HCCL_OVER_TCP=1 \
        -x LOG_LEVEL_ALL=3 \
    python bin/exec.py \
        --config-name=a21 \
        mode=train \
        run.id=${name} \
        framework=torch \
        run.compute_mode=HPU \
        run.distributed=False \
        data=real \
        data.data_directory=/lambda_stor/data/datascience/cosmic_tagging/ \
        run.precision=3
done
