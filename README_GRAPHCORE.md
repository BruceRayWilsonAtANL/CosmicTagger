# CosmicTagger on Graphcore

## Set Up

This is the contents of gc_ct_31.sh.

```bash

if [ "${1}" == "install" ] ; then
    mkdir -p ~/venvs/graphcore
    rm -rf ~/venvs/graphcore/cosmictagger31_env
    virtualenv ~/venvs/graphcore/cosmictagger31_env
fi

source ~/venvs/graphcore/cosmictagger31_env/bin/activate

if [ "${1}" == "install" ] ; then
    POPLAR_SDK_ROOT=/software/graphcore/poplar_sdk/3.1.0
    export POPLAR_SDK_ROOT=$POPLAR_SDK_ROOT
    pip install $POPLAR_SDK_ROOT/poptorch-3.1.0+98660_0a383de63f_ubuntu_20_04-cp38-cp38-linux_x86_64.whl
    mkdir ~/tmp
fi

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
fi

git checkout Graphcore
#git checkout GraphcoreDDP
```

## Master Shell Script

This is the contents of master_nvme.sh:

```bash
rm -rf outputs
rm bfloat16_*.log
source ipu_test_popdist_64.sh > bfloat16_2x10_64_nvme.log 2>&1
source ipu_test_popdist_1.sh > bfloat16_2x10_1_nvme.log 2>&1
source ipu_test_popdist_2.sh > bfloat16_2x10_2_nvme.log 2>&1
source ipu_test_popdist_4.sh > bfloat16_2x10_4_nvme.log 2>&1
source ipu_test_popdist_8.sh > bfloat16_2x10_8_nvme.log 2>&1
source ipu_test_popdist_16.sh > bfloat16_2x10_16_nvme.log 2>&1
```

## Scaling Results

For each quantity of IPUs, i.e., 1, 2, 4, 8, 16, and 64:

### Upadate bin/analyze_profiles

These are the lines to be updated:

```python
date      = '2023-02-15'
time      = '11-18-20'
config    = 'A21'
folderStr = "bfloat16_2x10_16/"
log_top = pathlib.Path(f"/home/wilsonb/DL/github.com/BruceRayWilsonAtANL/CosmicTagger/outputs/{date}/{time}/output/torch/{config}/")
```

### Process Log

```bash
python3 bin/analyze_profiles.py
python3 bin/analyze_profiles.py >> results.log
cat results.log
```

## Results

There are two 64 IPU results.

```console
/home/wilsonb/DL/github.com/BruceRayWilsonAtANL/CosmicTagger/outputs/2023-02-14/23-16-39/output/torch/A21/bfloat16_2x10_64/process.log
IO:
  Mean time: 0.029 +/- 0.007
  Median time: 0.028
  Max time: 0.080 (50)
  50/75/90/95 quantile: [0.028 0.031 0.036 0.041]
Step:
  Mean time: 0.087 +/- 0.024
  Median time: 0.083
  Max time: 0.340 (49)
  50/75/90/95 quantile: [0.083 0.093 0.11  0.11 ]
Forward: (estimated)
  Mean time: 3.385 +/- 60.032
  Median time: 22.609
  Max time: 38.575 (64)
  50/75/90/95 quantile: [22.60899925 24.02599907 27.7564003  30.45380058]
Durations:
  Mean time: 3.385 +/- 60.032
  Median time: 22.609
  Max time: 38.575 (64)
  50/75/90/95 quantile: [22.60899925 24.02599907 27.7564003  30.45380058]
FOM:  None
/home/wilsonb/DL/github.com/BruceRayWilsonAtANL/CosmicTagger/outputs/2023-02-14/23-16-40/output/torch/A21/bfloat16_2x10_64/process.log
IO:
  Mean time: 0.032 +/- 0.009
  Median time: 0.031
  Max time: 0.076 (438)
  50/75/90/95 quantile: [0.031 0.036 0.04  0.045]
Step:
  Mean time: 0.104 +/- 0.032
  Median time: 0.100
  Max time: 0.290 (428)
  50/75/90/95 quantile: [0.1  0.11 0.13 0.15]
Forward: (estimated)
  Mean time: 0.600 +/- 61.396
  Median time: 22.596
  Max time: 37.829 (488)
  50/75/90/95 quantile: [22.59600067 23.87999916 30.42680016 31.0182003 ]
Durations:
  Mean time: 0.600 +/- 61.396
  Median time: 22.596
  Max time: 37.829 (488)
  50/75/90/95 quantile: [22.59600067 23.87999916 30.42680016 31.0182003 ]
FOM:  None
/home/wilsonb/DL/github.com/BruceRayWilsonAtANL/CosmicTagger/outputs/2023-02-15/03-45-12/output/torch/A21/bfloat16_2x10_1/process.log
IO:
  Mean time: 0.016 +/- 0.003
  Median time: 0.015
  Max time: 0.033 (338)
  50/75/90/95 quantile: [0.015   0.017   0.019   0.02065]
Step:
  Mean time: 0.043 +/- 0.005
  Median time: 0.045
  Max time: 0.059 (470)
  50/75/90/95 quantile: [0.045 0.047 0.05  0.051]
Forward: (estimated)
  Mean time: 13.714 +/- 1.522
  Median time: 13.100
  Max time: 19.838 (38)
  50/75/90/95 quantile: [13.09950018 13.7327497  15.18330011 17.87894974]
Durations:
  Mean time: 13.714 +/- 1.522
  Median time: 13.100
  Max time: 19.838 (38)
  50/75/90/95 quantile: [13.09950018 13.7327497  15.18330011 17.87894974]
FOM:  0.1458
/home/wilsonb/DL/github.com/BruceRayWilsonAtANL/CosmicTagger/outputs/2023-02-15/05-39-57/output/torch/A21/bfloat16_2x10_2/process.log
IO:
  Mean time: 0.015 +/- 0.004
  Median time: 0.014
  Max time: 0.053 (406)
  50/75/90/95 quantile: [0.014 0.015 0.016 0.018]
Step:
  Mean time: 0.040 +/- 0.002
  Median time: 0.039
  Max time: 0.047 (1)
  50/75/90/95 quantile: [0.039  0.042  0.0432 0.044 ]
Forward: (estimated)
  Mean time: 6.583 +/- 22.157
  Median time: 12.841
  Max time: 18.020 (190)
  50/75/90/95 quantile: [12.8409996  12.98099995 13.11739979 17.60479965]
Durations:
  Mean time: 6.583 +/- 22.157
  Median time: 12.841
  Max time: 18.020 (190)
  50/75/90/95 quantile: [12.8409996  12.98099995 13.11739979 17.60479965]
FOM:  None
/home/wilsonb/DL/github.com/BruceRayWilsonAtANL/CosmicTagger/outputs/2023-02-15/07-32-08/output/torch/A21/bfloat16_2x10_4/process.log
IO:
  Mean time: 0.015 +/- 0.003
  Median time: 0.014
  Max time: 0.060 (399)
  50/75/90/95 quantile: [0.014 0.015 0.017 0.018]
Step:
  Mean time: 0.040 +/- 0.003
  Median time: 0.040
  Max time: 0.048 (256)
  50/75/90/95 quantile: [0.04  0.042 0.044 0.045]
Forward: (estimated)
  Mean time: 3.154 +/- 32.749
  Median time: 12.907
  Max time: 18.143 (246)
  50/75/90/95 quantile: [12.90699959 13.01599979 13.20919971 17.74379997]
Durations:
  Mean time: 3.154 +/- 32.749
  Median time: 12.907
  Max time: 18.143 (246)
  50/75/90/95 quantile: [12.90699959 13.01599979 13.20919971 17.74379997]
FOM:  None
/home/wilsonb/DL/github.com/BruceRayWilsonAtANL/CosmicTagger/outputs/2023-02-15/09-25-19/output/torch/A21/bfloat16_2x10_8/process.log
IO:
  Mean time: 0.015 +/- 0.003
  Median time: 0.014
  Max time: 0.057 (339)
  50/75/90/95 quantile: [0.014 0.015 0.017 0.018]
Step:
  Mean time: 0.041 +/- 0.003
  Median time: 0.040
  Max time: 0.052 (2)
  50/75/90/95 quantile: [0.04  0.042 0.045 0.046]
Forward: (estimated)
  Mean time: 1.759 +/- 35.239
  Median time: 12.939
  Max time: 18.124 (75)
  50/75/90/95 quantile: [12.93900013 13.04199982 13.17800045 17.68339996]
Durations:
  Mean time: 1.759 +/- 35.239
  Median time: 12.939
  Max time: 18.124 (75)
  50/75/90/95 quantile: [12.93900013 13.04199982 13.17800045 17.68339996]
FOM:  None
/home/wilsonb/DL/github.com/BruceRayWilsonAtANL/CosmicTagger/outputs/2023-02-15/11-18-20/output/torch/A21/bfloat16_2x10_16/process.log
IO:
  Mean time: 0.015 +/- 0.003
  Median time: 0.015
  Max time: 0.046 (410)
  50/75/90/95 quantile: [0.015 0.016 0.018 0.019]
Step:
  Mean time: 0.043 +/- 0.004
  Median time: 0.042
  Max time: 0.057 (16)
  50/75/90/95 quantile: [0.042 0.046 0.049 0.051]
Forward: (estimated)
  Mean time: 0.890 +/- 37.822
  Median time: 13.584
  Max time: 18.888 (136)
  50/75/90/95 quantile: [13.58399963 13.69999981 13.83459969 14.13099976]
Durations:
  Mean time: 0.890 +/- 37.822
  Median time: 13.584
  Max time: 18.888 (136)
  50/75/90/95 quantile: [13.58399963 13.69999981 13.83459969 14.13099976]
FOM:  None
```
