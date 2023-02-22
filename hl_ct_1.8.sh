# rm -rf ~/.local
if [ "${1}" == "install" ] ; then
    rm -rf ~/venvs/habana/venv_ct
    python3 -m venv --system-site-packages ~/venvs/habana/venv_ct
fi

source ~/venvs/habana/venv_ct/bin/activate

cd ~/DL/github.com/BruceRayWilsonAtANL/CosmicTagger_habana
export PYTHON=$(which python)
export PYTHONPATH=/home/wilsonb/DL/Habana/Model-References/:$(which python)
#cd ~/CosmicTagger
#export PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING=false
export HABANA_LOGS=~/.habana_logs
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu
export MPI_ROOT=/usr/local/openmpi
export MASTER_PORT=12355
# The public network is 140.221.77.101
# The private network is 192.168.201.0/24
export MASTER_ADDR=192.168.201.101

# This must work.
# ssh habana-02
# ssh habana-01
# exit
# exit

if [ "${1}" == "install" ] ; then
    pip install cmake scikit-build numpy h5py Pillow tensorboardX hydra-core==1.1
    pip install larcv
fi

git checkout Habana_1.8.0_redux
