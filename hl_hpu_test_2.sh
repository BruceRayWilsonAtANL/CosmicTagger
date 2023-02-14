python bin/exec.py \
  mode=train \
  run.id=200 \
  framework=torch \
  run.compute_mode=HPU \
  run.distributed=False \
  data.data_directory=/home/ac.ltran/CosmicTagger/example_data/ \
  data.file=cosmic_tagging_light.h5 \
  data.aux_file=cosmic_tagging_light.h5
