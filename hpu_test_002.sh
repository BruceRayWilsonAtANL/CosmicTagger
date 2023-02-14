#!/bin/bash
# git checkout Graphcore
for i in {1,}
do
    export name=bfloat16_2x10_${i}
    mpirun \
    -n 2 \
    --bind-to core \
    --map-by slot:PE=7 \
    --rank-by core \
    --report-bindings \
    --allow-run-as-root \
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
