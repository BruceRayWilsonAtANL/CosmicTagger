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
rm -rf ~/venvs/habana/venv_ct6
python3.8 -m venv --system-site-packages ~/venvs/habana/venv_ct6
source ~/venvs/habana/venv_ct6/bin/activate
pip3 install scikit-build numpy
```

### Habana

#### Install Requirements System Admin Only

```bash
deactive
sudo ./tensorflow_system_installation.sh
sudo ./pytorch_system_installation.sh
source ~/venvs/habana/venv_ct6/bin/activate
```

#### Install Requirements User

**NOTE: There will be some warnings for packages not found.
Ignore them**

```bash
source ~/venvs/habana/venv_ct6/bin/activate
cd ~/DL/BruceRayWilsonAtANL/CosmicTagger
python3 -m pip install scikit-build numpy
python3 -m pip install -r requirements.txt
```

```bash
export PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING=false
export HABANA_LOGS=~/.habana_logs
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu
```

Example:

```bash
xpython bin/exec.py mode=train run.id='02' run.distributed=False data.data_directory=/lambda_stor/data/datascience/cosmic_tagging/ framework=torch network.normalization=batch run.minibatch_size=2
```

```bash
xpython bin/exec.py mode=train run.id='02' run.distributed=False data.data_directory=/lambda_stor/data/datascience/cosmic_tagging/ framework=torch network.normalization=batch run.minibatch_size=2 run.iterations=1
```

```bash
xpython bin/exec.py mode=train run.id='02' run.distributed=False data.data_directory=/lambda_stor/data/datascience/cosmic_tagging/ framework=torch run.minibatch_size=2 run.iterations=1
```

From Corey Adams:

```text
n_epochs = run.iterations * run.minibatch_size / 43075
Where 43075 is the dataset size.
```

```text
To get SoTA performance you need batch-size ~ 64, which you can replicate here with gradient accumulation if habana supports it.  So, you’d be looking at 2s/Image * 64 = 2min per iteration.  Min 3000+ iterations = 4+ days
```

```text
Try mode.optimizer.gradient_accumulation=N
Where N is how many batches you want to accumulate
```

```text
# For half precision, we disable gradient accumulation.  This is to allow
# dynamic loss scaling
```

I, Bruce, asked Habana if they support gradient accumulation and they said yes.

```bash
screen
git checkout Habana_1.5.0
source fp32_1_500.sh
source fp16_1_500.sh
```

## Multi-HPU Runs

```bash
# Does not work.
mpirun -n 8 --bind-to core --map-by slot:PE=7 --rank-by core --report-bindings --allow-run-as-root \
python bin/exec.py \
    mode=train \
    run.id='fp16_10x1' \
    run.distributed=True \
    data.data_directory=/lambda_stor/data/datascience/cosmic_tagging/ \
    framework=torch \
    run.compute_mode=HPU \
    run.minibatch_size=1 \
    run.iterations=10 \
    run.precision=3

# Trying...  Starts well...  Ends well.
mpirun -n 8 --bind-to core --map-by slot:PE=7 --rank-by core --report-bindings --allow-run-as-root \
python bin/exec.py \
    mode=train \
    run.id='fp16_10x1' \
    run.distributed=True \
    data.data_directory=/lambda_stor/data/datascience/cosmic_tagging/ \
    framework=torch \
    run.compute_mode=HPU \
    run.minibatch_size=8 \
    run.iterations=10 \
    run.precision=3

# Try
mpirun -n 8 --bind-to core --map-by slot:PE=7 --rank-by core --report-bindings --allow-run-as-root \
python bin/exec.py \
    mode=train \
    run.id='fp32_2x500' \
    run.distributed=True \
    data.data_directory=/lambda_stor/data/datascience/cosmic_tagging/ \
    framework=torch \
    run.compute_mode=HPU \
    run.minibatch_size=16 \
    run.iterations=500 \
    run.precision=0
```

## My Notes

```text
For multi-node please check the doc https://docs.habana.ai/en/latest/PyTorch/PyTorch_Scaling_Guide/DDP_Based_Scaling.html#scale-out-using-gaudi-nics
How resnet work on multiple node is an example you can follow: https://github.com/HabanaAI/Model-References/tree/master/PyTorch/computer_vision/classification/torchvision#multinode-training
```

```text
You can directly set -x GC_KERNEL_PATH without explicitly setting it, and it will inherit the value from current environment. For URL I guess you mean the domain name, it should work for -H , but it might not work for --mca btl_tcp_if_include. You can give it a try. If it doesn't work you can use ip -br -4 a  to find out the IP of current machine.
```
