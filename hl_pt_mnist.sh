#!/bin/bash
set -e
cd
rm -rf ~/.local
python3 -m venv --system-site-packages ~/PT_venv
source ~/PT_venv/bin/activate
export PYTHON=$(which python)
export HABANA_LOGS=~/.habana_logs
export PYTHONPATH=~/Model-References:$(which python)
cd ~/Model-References/PyTorch/examples/computer_vision/hello_world
for i in {1, 2, 3, 4. 5. 6,}
do
    echo {$i}
    python mnist.py --hpu
    python mnist.py --hpu --hmp
    python mnist.py --hpu --use_lazy_mode
    python mnist.py --hpu --hmp --use_lazy_mode
    mpirun -n 4 --bind-to core --map-by slot:PE=7 --rank-by core --report-bindings --allow-run-as-root python3 mnist.py --batch-size=64 --epochs=1 --lr=1.0 --gamma=0.7 --hpu --hmp --hmp-bf16=ops_bf16_mnist.txt --hmp-fp32=ops_fp32_mnist.txt
    mpirun -n 8 --bind-to core --map-by slot:PE=7 --rank-by core --report-bindings --allow-run-as-root python3 mnist.py --batch-size=64 --epochs=1 --lr=1.0 --gamma=0.7 --hpu --hmp --hmp-bf16=ops_bf16_mnist.txt --hmp-fp32=ops_fp32_mnist.txt
done
