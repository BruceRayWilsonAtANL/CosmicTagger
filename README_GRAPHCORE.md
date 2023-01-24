# CosmicTagger on Graphcore

## Set Up

This is the contents of gc_ct_31.sh.

```bash
#mkdir -p ~/venvs/graphcore
rm -rf ~/venvs/graphcore/cosmictagger31_env
virtualenv ~/venvs/graphcore/cosmictagger31_env
source ~/venvs/graphcore/cosmictagger31_env/bin/activate
POPLAR_SDK_ROOT=/software/graphcore/poplar_sdk/3.1.0
export POPLAR_SDK_ROOT=$POPLAR_SDK_ROOT
pip install $POPLAR_SDK_ROOT/poptorch-3.1.0+98660_0a383de63f_ubuntu_20_04-cp38-cp38-linux_x86_64.whl
mkdir ~/tmp
export TF_POPLAR_FLAGS=--executable_cache_path=~/tmp
export PYTORCH_CACHE_DIR=~/tmp

export POPART_LOG_LEVEL=WARN
export POPLAR_LOG_LEVEL=WARN
export POPLIBS_LOG_LEVEL=WARN

export PYTHONPATH=/software/graphcore/poplar_sdk/3.1.0/poplar-ubuntu_20_04-3.1.0+6824-9c103dc348/python:$PYTHONPATH
cd ~/DL/BruceRayWilsonAtANL/CosmicTagger
python3 -m pip install scikit-build numpy
python3 -m pip install -r requirements.txt
git checkout GraphcoreDDP
```

## Run

This is the contents of ipu_test_popdist_1.sh that can be used to run the code.

```bash
#!/bin/bash
# git checkout GraphcoreDDP
for i in {1,}
do
    name=bfloat16_2x10_${i}
    poprun \
    -vv \
    --num-instances=4 \
    --num-replicas=4 \
    --executable-cache-path=$PYTORCH_CACHE_DIR \
    python bin/exec.py \
    mode=train \
    run.id=${name} \
    run.distributed=False \
    data.data_directory=/lambda_stor/data/datascience/cosmic_tagging/ \
    framework=torch \
    run.compute_mode=IPU \
    run.minibatch_size=2 \
    run.iterations=10 \
    run.precision=3
done
```