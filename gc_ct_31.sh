
if [ "${1}" == "install" ] ; then
    mkdir -p ~/venvs/graphcore
    rm -rf ~/venvs/graphcore/cosmictagger31_env
    virtualenv ~/venvs/graphcore/cosmictagger31_env

source ~/venvs/graphcore/cosmictagger31_env/bin/activate

if [ "${1}" == "install" ] ; then
    POPLAR_SDK_ROOT=/software/graphcore/poplar_sdk/3.1.0
    export POPLAR_SDK_ROOT=$POPLAR_SDK_ROOT
    pip install $POPLAR_SDK_ROOT/poptorch-3.1.0+98660_0a383de63f_ubuntu_20_04-cp38-cp38-linux_x86_64.whl
    mkdir ~/tmp

export TF_POPLAR_FLAGS=--executable_cache_path=~/tmp
export POPTORCH_CACHE_DIR=~/tmp
export PYTORCH_CACHE_DIR=~/tmp

export POPART_LOG_LEVEL=WARN
export POPLAR_LOG_LEVEL=WARN
export POPLIBS_LOG_LEVEL=WARN

export PYTHONPATH=/software/graphcore/poplar_sdk/3.1.0/poplar-ubuntu_20_04-3.1.0+6824-9c103dc348/python:$PYTHONPATH
cd ~/DL/github.com/BruceRayWilsonAtANL/CosmicTagger

if [ "${1}" == "install" ] ; then
    python3 -m pip install scikit-build numpy
    python3 -m pip install -r requirements.txt

git checkout Graphcore
#git checkout GraphcoreDDP
