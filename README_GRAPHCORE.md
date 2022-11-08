# CosmicTagger on Graphcore

## From Habana

```bash
python bin/exec.py \
    mode=train \
    mode.optimizer.gradient_accumulation=10 \
    run.id='fp32_10x2_1' \
    run.distributed=False \
    data.data_directory=/lambda_stor/data/datascience/cosmic_tagging/ \
    framework=torch \
    run.compute_mode=HPU \
    run.minibatch_size=2 \
    run.iterations=10 \
    run.precision=0 > fp32_10x2_1.log 2>&1 &
```

```bash
for i in {1..5}
do
    name=bfloat16_2x10_${i}
    python bin/exec.py \
    mode=train \
    run.id=${name} \
    run.distributed=False \
    data.data_directory=/lambda_stor/data/datascience/cosmic_tagging/ \
    framework=torch \
    run.compute_mode=HPU \
    run.minibatch_size=2 \
    run.iterations=10 \
    run.precision=2 \
    > ${name}.log 2>&1 &
done
```

## Graphcore

```bash
for i in {1}
do
    name=bfloat16_2x10_${i}
    python bin/exec.py \
    mode=train \
    run.id=${name} \
    run.distributed=False \
    data.data_directory=/lambda_stor/data/datascience/cosmic_tagging/ \
    framework=torch \
    run.compute_mode=HPU \
    run.minibatch_size=2 \
    run.iterations=10 \
    run.precision=2 \
    > ${name}.log 2>&1 &
done
```