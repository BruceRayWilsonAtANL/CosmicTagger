for i in {1..5}
do
    name=mixed_2x10_${i}
    python bin/exec.py \
    mode=train \
    run.id=${name} \
    run.distributed=False \
    data.data_directory=/lambda_stor/data/datascience/cosmic_tagging/ \
    framework=torch \
    run.compute_mode=HPU \
    run.minibatch_size=2 \
    run.iterations=10 \
    run.precision=1 \
    --hmp --hmp-bf16=ops_bf16_mnist.txt --hmp-fp32=ops_fp32_mnist.txt > ${name}.log 2>&1 &
