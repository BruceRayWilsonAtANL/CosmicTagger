#!/bin/bash
# git checkout GraphcoreDDP
for i in {1,}
do
    name=bfloat16_2x10_${i}
    poprun \
    -vv \
    --num-instances=4 \
    --num-replicas=4 \
    --executable-cache-path=$PYTORCH_CACHE_DIR \
    python bin/exec.py \
    mode=train \
    run.id=${name} \
    run.distributed=False \
    data.data_directory=/lambda_stor/data/datascience/cosmic_tagging/ \
    framework=torch \
    run.compute_mode=IPU \
    run.minibatch_size=2 \
    run.iterations=10 \
    run.precision=3 \
    > ${name}.log 2>&1 &
done
