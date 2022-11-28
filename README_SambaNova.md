# README SambaNova

```bash
/opt/sambaflow/apps/private/anl/cosmictagger.py \
    compile \
    --residual \
    -b 2 \
    --enable-tiling=True \
    --mac-v1 \
    --mac-human-decision \
    /opt/sambaflow/apps/private/anl/cosmictagger/human_decisions_tiled.json \
    --compiler-configs /opt/sambaflow/apps/private/anl/cosmictagger/ \
    compiler_configs_generator.json \
    --pef-name=cosmictagger \
    --output-folder="$HOME/out"
python /opt/sambaflow/apps/private/anl/cosmictagger.py \
    run \
    --residual \
    -b 2 \
    --enable-tiling=True \
    -f /var/tmp/dataset/cosmictagger/cosmic_tagging_train.h5 \
    -lr 0.0003 \
    -i 200 \
    --acc-test \
    --pef=“$HOME/out/cosmictagger/cosmictagger.pef” \
    --mac-v1
```
