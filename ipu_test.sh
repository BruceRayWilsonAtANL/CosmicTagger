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
    run.precision=3 \
    > ${name}.log 2>&1 &
done


python3.8 bin/exec.py \
mode=train \
run.id=01 \
run.distributed=False \
data.data_directory=/lambda_stor/data/datascience/cosmic_tagging/ \
framework=torch \
run.compute_mode=IPU \
run.minibatch_size=1 \
run.iterations=1 \
run.precision=3
