#!/bin/bash
# git checkout Graphcore
for i in {1}
do
    name=bfloat16_2x10_${i}
    python bin/exec.py \
    mode=train \
    run.id=${name} \
    run.distributed=False \
    data.data_directory=/lambda_stor/data/datascience/cosmic_tagging/ \
    framework=torch \
    run.compute_mode=CPU \
    run.minibatch_size=1 \
    run.iterations=1 \
    run.precision=2 \
    > ${name}.log 2>&1 &
done
