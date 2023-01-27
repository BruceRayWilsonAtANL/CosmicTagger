#! /bin/bash
export SF_RNT_LOG_LEVEL=DEBUG
LOGDIR=`date +%m%d%y.%H`
MODEL_NAME="cosmic_tagger"
OUTPUT_PATH=/data/ANL/results/$(hostname)/${USER}/${LOGDIR}/${MODEL_NAME}.out
echo "Using ${OUTPUT_PATH} for output"
mkdir -p /data/ANL/results/$(hostname)/${USER}/${LOGDIR}

export PYTHONPATH=/home/wilsonb/DL/github.com/BruceRayWilsonAtANL/CosmicTagger

#######################
# Edit these variables.
#######################
export OMP_NUM_THREADS=4
source /opt/sambaflow/apps/private/anl/venv/bin/activate

#######################
# Start script timer
SECONDS=0
# Temp file location
DIRECTORY=cosmictagger
OUTDIR=/nvmedata/scratch/wilsonb/${MODEL_NAME}
if [ ! -d ${OUTDIR} ] ; then
  mkdir -p ${OUTDIR}
fi
cd ${HOME}
echo "Model: ${MODEL_NAME}" > ${OUTPUT_PATH} 2>&1
echo "Date: " $(date +%m/%d/%y) >> ${OUTPUT_PATH} 2>&1
echo "Time: " $(date +%H:%M) >> ${OUTPUT_PATH} 2>&1

echo "Machine State Before: " >> ${OUTPUT_PATH} 2>&1
/opt/sambaflow/bin/snfadm -l inventory >> ${OUTPUT_PATH} 2>&1

if [ ! -e ${OUTDIR}/cosmictagger/cosmictagger.pef ] ; then
  echo "COMPILE START AT ${SECONDS}" >> ${OUTPUT_PATH} 2>&1
  COMMAND="python -m pdb /home/wilsonb/DL/github.com/BruceRayWilsonAtANL/CosmicTagger/bin/exec.py \
    --enable-stoc-rounding --enable-conv-tiling --compiler-configs-file /opt/sambaflow/apps/private/anl/cosmictagger/jsons/compiler_configs/compiler_configs.json --mac-human-decision /opt/sambaflow/apps/private/anl/cosmictagger/jsons/hd_files/cosmic_uresnet3d.json --residual --batch-size 4 --data-parallel -ws 2 --num-tiles=4 --model-type uresnet3d --downsampling convolutional \
    --upsampling convolutional \
    --connections concat \
    --run-benchmark \
    --pef-name=cosmictagger \
    --output-folder=${OUTDIR} \
    --blocks-final 5" \
    overrides \
    mode=train \
    run.id=${name} \
    run.distributed=False \
    data.data_directory=/lambda_stor/data/datascience/cosmic_tagging/ \
    data.downsample=0 \
    framework=torch \
    run.compute_mode=CPU \
    run.minibatch_size=1 \
    run.iterations=1 \
    run.precision=3

  echo "COMPILE COMMAND: $COMMAND" >> ${OUTPUT_PATH} 2>&1
  eval $COMMAND
  echo "COMPILE END AT ${SECONDS}" >> ${OUTPUT_PATH} 2>&1
fi
