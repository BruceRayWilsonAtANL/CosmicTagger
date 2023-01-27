#! /bin/bash
#set -e
export SF_RNT_LOG_LEVEL=DEBUG
LOGDIR=`date +%m%d%y.%H`
if [ "$1" ] ; then
LOGDIR=$1
fi
MODEL_NAME="cosmic_tagger"
OUTPUT_PATH=/data/ANL/results/$(hostname)/${USER}/${LOGDIR}/${MODEL_NAME}.out
echo "Using ${OUTPUT_PATH} for output"
mkdir -p /data/ANL/results/$(hostname)/${USER}/${LOGDIR}


#######################
# Edit these variables.
#######################
export OMP_NUM_THREADS=4
source /opt/sambaflow/apps/private/anl/venv/bin/activate
#source ~/venvs/sambanova/cosmictagger_env/bin/activate
export PYTHONPATH=/home/wilsonb/DL/github.com/BruceRayWilsonAtANL/CosmicTagger

#######################
# Start script timer
SECONDS=0
# Temp file location
DIRECTORY=cosmictagger
OUTDIR=${HOME}/${MODEL_NAME}
if [ ! -d ${OUTDIR} ] ; then
  mkdir ${OUTDIR}
fi
cd ${HOME}
echo "Model: ${MODEL_NAME}" > ${OUTPUT_PATH} 2>&1
echo "Date: " $(date +%m/%d/%y) >> ${OUTPUT_PATH} 2>&1
echo "Time: " $(date +%H:%M) >> ${OUTPUT_PATH} 2>&1

echo "Machine State Before: " >> ${OUTPUT_PATH} 2>&1
/opt/sambaflow/bin/snfadm -l inventory >> ${OUTPUT_PATH} 2>&1

if [ ! -e ${OUTDIR}/cosmictagger/cosmictagger.pef ] ; then
  echo "COMPILE START AT ${SECONDS}" >> ${OUTPUT_PATH} 2>&1
  COMMAND="python /home/wilsonb/DL/github.com/BruceRayWilsonAtANL/CosmicTagger/bin/exec.py compile +enable-stoc-rounding +enable-conv-tiling +compiler-configs-file /opt/sambaflow/apps/private/anl/cosmictagger/jsons/compiler_configs/compiler_configs.json +mac-human-decision /opt/sambaflow/apps/private/anl/cosmictagger/jsons/hd_files/cosmic_uresnet3d.json +residual +batch-size 4 +data-parallel -ws 2 +num-tiles=4 +model-type uresnet3d +downsampling convolutional +upsampling convolutional +connections concat +run-benchmark +pef-name=cosmictagger +output-folder=${OUTDIR} +blocks-final 5"

  echo "COMPILE COMMAND: $COMMAND" >> ${OUTPUT_PATH} 2>&1
  eval $COMMAND >> ${OUTPUT_PATH} 2>&1
  echo "COMPILE END AT ${SECONDS}" >> ${OUTPUT_PATH} 2>&1
fi
