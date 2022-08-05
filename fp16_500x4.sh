for i in {1..5}
do
    name=float16_4x500_${i}
    python bin/exec.py \
    mode=train \
    run.id=${name} \
    run.distributed=False \
    data.data_directory=/lambda_stor/data/datascience/cosmic_tagging/ \
    framework=torch \
    run.compute_mode=HPU \
    run.minibatch_size=4 \
    run.iterations=500 \
    run.precision=3 > ${name}.log 2>&1 &
done
