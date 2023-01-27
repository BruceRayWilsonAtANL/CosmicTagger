#!/bin/bash -x
export ONM_NUM_THREADS=8
SECONDS=0
if [ "${1}" == "compile" ] ; then
  python ./bin/exec.py compile --residual -b 2 --enable-tiling=True --mac-v1 --mac-human-decision /opt/sambaflow/apps/private/anl/cosmictagger/human_decisions_tiled.json --compiler-configs /opt/sambaflow/apps/private/anl/cosmictagger/compiler_configs_generator.json --pef-name=cosmictagger --output-folder="$HOME/$(hostname)/out"

elif [ "${1}" == "run" ] ; then
srun --gres=rdu:1 python ./bin/exec.py run --residual -b 2 --enable-tiling=True -f /var/tmp/dataset/cosmictagger/cosmic_tagging_train.h5 -lr 0.0003 -i 200 --acc-test --pef="$HOME/$(hostname)/out/cosmictagger/cosmictagger.pef" --mac-v1

elif [ "${1}" == "mp" ] ; then
srun --gres=rdu:1 python ./bin/exec.py measure-performance --residual -b 2 --enable-tiling=True -f /var/tmp/dataset/cosmictagger/cosmic_tagging_train.h5 -lr 0.0003 -i 200 --acc-test --pef="$HOME/$(hostname)/out/cosmictagger/cosmictagger.pef" --num-iterations 100 --mac-v1

fi
