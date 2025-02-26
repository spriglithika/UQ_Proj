#!/bin/bash
#
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --time 5:00:00
#SBATCH --output /proj/agp/users/%u/logs/%j.out
#SBATCH --account=berzelius-2023-365
#SBATCH --job-name=vad_eval
#

CONFIG=$1
CHECKPOINT=$2
GPUS=1

MASTER_PORT=${MASTER_PORT:-28596}
GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))


singularity exec --nv \
    --bind $PWD:/vad \
    --bind /proj:/proj \
    --bind /proj/adas-data/data/nuscenes:/vad/data/nuscenes \
    --pwd /vad \
    --env PYTHONPATH="/vad:${PYTHONPATH}" \
    /proj/agp/containers/vad-21-02-2024.sif \
    python -m torch.distributed.launch \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_port=$MASTER_PORT \
    tools/test.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher pytorch ${@:3} \
    --eval bbox \