# rm -rf ~/.local
rm -rf ~/venvs/habana/venv_ct
python3 -m venv --system-site-packages ~/venvs/habana/venv_ct
source ~/venvs/habana/venv_ct/bin/activate

cd ~/DL/github.com/BruceRayWilsonAtANL/CosmicTagger_habana
#cd ~/CosmicTagger
#export PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING=false
export HABANA_LOGS=~/.habana_logs
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu
export MPI_ROOT=/usr/local/openmpi
pip install cmake scikit-build numpy h5py Pillow tensorboardX hydra-core==1.1
pip install larcv
