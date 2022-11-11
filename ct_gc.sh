#!/bin/bash
# exit when any command fails
#set -e
source /lambda_stor/software/graphcore/poplar_sdk/3.0.0/popart-ubuntu_20_04-3.0.0+5691-1e179b3b85/enable.sh
source /lambda_stor/software/graphcore/poplar_sdk/3.0.0/poplar-ubuntu_20_04-3.0.0+5691-1e179b3b85/enable.sh
mkdir -p ~/venvs/graphcore
rm -rf ~/venvs/graphcore/cosmictagger_env
virtualenv ~/venvs/graphcore/cosmictagger_env
source ~/venvs/graphcore/cosmictagger_env/bin/activate
POPLAR_SDK_ROOT=/lambda_stor/software/graphcore/poplar_sdk/3.0.0
export POPLAR_SDK_ROOT=$POPLAR_SDK_ROOT
pip install $POPLAR_SDK_ROOT/poptorch-3.0.0+86945_163b7ce462_ubuntu_20_04-cp38-cp38-linux_x86_64.whl
#mkdir ~/tmp
export TF_POPLAR_FLAGS=--executable_cache_path=~/tmp
export POPTORCH_CACHE_DIR=~/tmp
export POPART_LOG_LEVEL=WARN
export POPLAR_LOG_LEVEL=WARN
export POPLIBS_LOG_LEVEL=WARN
export PYTHONPATH=/lambda_stor/software/graphcore/poplar_sdk/3.0.0/poplar-ubuntu_20_04-3.0.0+5691-1e179b3b85/python:$PYTHONPATH
cd ~/DL/BruceRayWilsonAtANL/CosmicTagger


python3 -m pip install scikit-build numpy
python3 -m pip install -r requirements.txt



#
# From Habana
#

deactivate
rm -rf ~/venvs/habana/venv_ct
python3 -m venv --system-site-packages ~/venvs/habana/venv_ct
source ~/venvs/habana/venv_ct/bin/activate
cd ~/DL/BruceRayWilsonAtANL/CosmicTagger
export PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING=false
export HABANA_LOGS=~/.habana_logs
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu
export MPI_ROOT=/usr/local/openmpi
python3 -m pip install scikit-build numpy
python3 -m pip install -r requirements.txt
pip install --upgrade habana-tensorflow habana-horovod habana-torch-dataloader habana-torch-plugin



