for i in {1..5}
do
    minibatch_size=1
    iterations=500
    name=fp32_${minibatch_size}x${iterations}_${i}
    python bin/exec.py \
    mode=train \
    run.id=${name} \
    run.distributed=False \
    data.data_directory=/lambda_stor/data/datascience/cosmic_tagging/ \
    framework=torch \
    run.compute_mode=HPU \
    run.minibatch_size=${minibatch_size} \
    run.iterations=${iterations} \
    run.precision=0 > ${name}.log 2>&1 &
done