#!/bin/bash
# git checkout Graphcore
python3.8 bin/exec.py \
--config-name=a21 \
mode=train \
run.id=01 \
run.distributed=False \
data=real \
data.data_directory=/lambda_stor/data/datascience/cosmic_tagging/ \
framework=torch \
run.compute_mode=CPU \
run.precision=3 \
hydra/job_logging=disabled
