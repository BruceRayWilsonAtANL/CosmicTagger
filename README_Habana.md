[![Build Status](https://travis-ci.com/coreyjadams/CosmicTagger.svg?branch=master)](https://travis-ci.com/coreyjadams/CosmicTagger)

# Neutrino and Cosmic Tagging with UNet

## Install and Run Scripts From Habana

They do a couple of items different from our documentation.

```bash
pip install scikit-build ;\
apt update ; \
apt-get install -y libhdf5-dev ;\
git clone https://github.com/DeepLearnPhysics/larcv3.git ;\
cd larcv3 ;\
git submodule update --init ;\
pip install -e . ;\
cd .. ;\
pip install -r requirements.txt
```

```bash
python3 bin/exec.py mode=train run.id=100 \
    framework=torch \
    run.compute_mode=HPU \
    run.distributed=False \
    data.data_directory=example_data/ \
    data.file=cosmic_tagging_light.h5 \
    data.aux_file=cosmic_tagging_light.h5
```

## Installation

```bash
sudo apt-get install cmake libhdf5-serial-dev python-dev
```

```bash
rm -rf ~/venvs/habana/venv_ct5
python3.8 -m venv --system-site-packages ~/venvs/habana/venv_ct5
source ~/venvs/habana/venv_ct5/bin/activate
pip3 install scikit-build numpy
```

### Habana

#### Install Requirements System Admin Only

```bash
# This must be done in the repo root.
deactive
sudo ./tensorflow_system_installation.sh
sudo ./pytorch_system_installation.sh
source ~/venvs/habana/venv_ct5/bin/activate
```

#### Install Requirements User

**NOTE: There will be some warnings for packages not found.
Ignore them**

```bash
# This must be done in the repo root.
# This might require sudo priviledges.
#PYTHON=`which python` ./pytorch_venv_installation.sh -sys
#PYTHON=`which python` ./tensorflow_venv_installation.sh --pip_user false
#source ~/.bashrc
source ~/venvs/habana/venv_ct5/bin/activate
export PYTHONPATH=$(which python)  # Maybe not.
pip install -r requirements.txt
```

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu
export PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING=false
export HABANA_LOGS=~/.habana_logs
```

Example:

```bash
python bin/exec.py mode=train run.id='02' run.distributed=False data.data_directory=/lambda_stor/data/datascience/cosmic_tagging/ framework=torch network.normalization=batch run.minibatch_size=2
```

```bash
python bin/exec.py mode=train run.id='02' run.distributed=False data.data_directory=/lambda_stor/data/datascience/cosmic_tagging/ framework=torch network.normalization=batch run.minibatch_size=2 run.iterations=1
```

```bash
python bin/exec.py mode=train run.id='02' run.distributed=False data.data_directory=/lambda_stor/data/datascience/cosmic_tagging/ framework=torch run.minibatch_size=2 run.iterations=1
```

```bash
screen
git checkout Habana_1.5.0
python bin/exec.py \
    mode=train \
    run.id='fp32 10x2' \
    run.distributed=False \
    data.data_directory=/lambda_stor/data/datascience/cosmic_tagging/ \
    framework=torch \
    run.compute_mode=HPU \
    run.minibatch_size=2 \
    run.iterations=10 \
    run.precision=0 > fp32_10x2_1.log 2>&1 &
```

```bash
screen
git checkout Habana_1.5.0
python bin/exec.py \
mode=train \
run.id='bf16 10x2' \
run.distributed=False \
data.data_directory=/lambda_stor/data/datascience/cosmic_tagging/ \
framework=torch \
run.minibatch_size=2 \
run.iterations=10 \
run.precision=2 > bf16_10x2_1.log 2>&1 &
```
