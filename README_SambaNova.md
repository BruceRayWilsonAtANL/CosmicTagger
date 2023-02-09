# CosmicTagger on SambaNova

Use Cosmic_tagger.sh from Rick W. in this directory.

## Run 1/24/23

Use Cosmic_tagger.sh in this directory but from Rick W.

## Set Up

```bash
mkdir -p ~/venvs/sambanova
rm -rf ~/venvs/sambanova/cosmictagger_env
virtualenv ~/venvs/sambanova/cosmictagger_env

source ~/venvs/sambanova/cosmictagger_env/bin/activate
mkdir ~/tmp
cd ~/DL/github.com/BruceRayWilsonAtANL/CosmicTagger
#
#
python3 -m pip install scikit-build numpy
python3 -m pip install -r requirements.txt
#git checkout SambaNova001
```

## Shell Script CPU

```bash
for i in {1,}
do
    name=bfloat16_2x10_${i}
    python bin/exec.py \
    mode=train \
    run.id=${name} \
    run.distributed=False \
    data.data_directory=/lambda_stor/data/datascience/cosmic_tagging/ \
    data.downsample=0 \
    framework=torch \
    run.compute_mode=CPU \
    run.minibatch_size=1 \
    run.iterations=1 \
    run.precision=3 \
    > ${name}.log 2>&1 &
done
```

## Shell Script IPU

```bash
for i in {1,}
do
    name=bfloat16_2x10_${i}
    python bin/exec.py \
    mode=train \
    run.id=${name} \
    run.distributed=False \
    data.data_directory=/lambda_stor/data/datascience/cosmic_tagging/ \
    data.downsample=0 \
    framework=torch \
    run.compute_mode=IPU \
    run.minibatch_size=2 \
    run.iterations=10 \
    run.precision=3 \
    > ${name}.log 2>&1 &
done
```









```bash
/opt/sambaflow/apps/private/anl/cosmictagger.py \
    compile \
    --residual \
    -b 2 \
    --enable-tiling=True \
    --mac-v1 \
    --mac-human-decision \
    /opt/sambaflow/apps/private/anl/cosmictagger/human_decisions_tiled.json \
    --compiler-configs /opt/sambaflow/apps/private/anl/cosmictagger/ \
    compiler_configs_generator.json \
    --pef-name=cosmictagger \
    --output-folder="$HOME/out"
python /opt/sambaflow/apps/private/anl/cosmictagger.py \
    run \
    --residual \
    -b 2 \
    --enable-tiling=True \
    -f /var/tmp/dataset/cosmictagger/cosmic_tagging_train.h5 \
    -lr 0.0003 \
    -i 200 \
    --acc-test \
    --pef=“$HOME/out/cosmictagger/cosmictagger.pef” \
    --mac-v1
```
