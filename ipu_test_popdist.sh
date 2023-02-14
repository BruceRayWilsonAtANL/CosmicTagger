#!/bin/bash
# git checkout Graphcore
for i in {1,}
do
    export name=bfloat16_2x10_${i}
    poprun \
    -vv \
    --num-instances=1 \
    --num-replicas=1 \
    --executable-cache-path=$PYTORCH_CACHE_DIR \
    python bin/exec.py \
    --config-name=a21 \
    mode=train \
    run.id=${name} \
    run.distributed=False \
    data=real \
    data.data_directory=/lambda_stor/data/datascience/cosmic_tagging/ \
    framework=torch \
    run.compute_mode=IPU \
    run.minibatch_size=2 \
    run.iterations=10 \
    run.precision=3
done
