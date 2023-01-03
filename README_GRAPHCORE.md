# CosmicTagger on Graphcore

## Set Up

```bash
source /software/graphcore/poplar_sdk/3.0.0/popart-ubuntu_20_04-3.0.0+5691-1e179b3b85/enable.sh
source /software/graphcore/poplar_sdk/3.0.0/poplar-ubuntu_20_04-3.0.0+5691-1e179b3b85/enable.sh
#mkdir -p ~/venvs/graphcore
#rm -rf ~/venvs/graphcore/cosmictagger_env
#virtualenv ~/venvs/graphcore/cosmictagger_env
source ~/venvs/graphcore/cosmictagger_env/bin/activate
POPLAR_SDK_ROOT=/software/graphcore/poplar_sdk/3.0.0
export POPLAR_SDK_ROOT=$POPLAR_SDK_ROOT
#pip install $POPLAR_SDK_ROOT/poptorch-3.0.0+86945_163b7ce462_ubuntu_20_04-cp38-cp38-linux_x86_64.whl
#mkdir ~/tmp
export TF_POPLAR_FLAGS=--executable_cache_path=~/tmp
export POPTORCH_CACHE_DIR=~/tmp
export POPART_LOG_LEVEL=WARN
export POPLAR_LOG_LEVEL=WARN
export POPLIBS_LOG_LEVEL=WARN
export PYTHONPATH=/software/graphcore/poplar_sdk/3.0.0/poplar-ubuntu_20_04-3.0.0+5691-1e179b3b85/python:$PYTHONPATH
cd ~/DL/BruceRayWilsonAtANL/CosmicTagger
#
#
#python3 -m pip install scikit-build numpy
#python3 -m pip install -r requirements.txt
#git checkout Graphcore
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
