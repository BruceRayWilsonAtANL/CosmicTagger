#!/bin/bash
# git checkout Graphcore
python3.8 bin/exec.py \
mode=train \
run.id=04x20_000 \
run.distributed=False \
data.data_directory=/lambda_stor/data/datascience/cosmic_tagging/ \
framework=torch \
run.compute_mode=IPU \
run.minibatch_size=4 \
run.iterations=20000 \
run.precision=3
