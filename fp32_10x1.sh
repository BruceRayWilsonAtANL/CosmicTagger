for i in {1..5}
do
    name=fp32_10x1_${i}
    python bin/exec.py \
    mode=train \
    run.id=${name} \
    run.distributed=False \
    data.data_directory=/lambda_stor/data/datascience/cosmic_tagging/ \
    framework=torch \
    run.compute_mode=HPU \
    run.minibatch_size=1 \
    run.iterations=500 \
    run.precision=0 > fp32_10x1_${i}.log 2>&1 &
done