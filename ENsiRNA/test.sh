#!/bin/bash

########## adjust configs according to your needs ##########
# usually only the DATA_DIR need to be revised
CODE_DIR=`realpath $(dirname "$0")/../..`
NUM_WORKERS=8
BATCH_SIZE=16
GPU="${GPU:-0}"  # using GPU 0 by default
CKPT=$1
TEST_SET=$2
SAVE_DIR="${3:-"$(dirname "$CKPT")/results"}"  # save to the same directory as the checkpoint by default
ID="${4:-result}" 

# validity check
if [ -z "$CKPT" ]; then
	#echo "Usage: bash $0 <checkpoint> <test set> [save_dir]"
	exit 1;
else
	CKPT=`realpath $CKPT`
	SAVE_DIR=`realpath $SAVE_DIR`
fi

# echo Configurations
#echo "Locate the project folder at ${CODE_DIR}"
#echo "Using GPU: ${GPU}"
#echo "Evaluating ${CKPT}"
#echo "Results will be written to ${SAVE_DIR}"

# set gpu
export CUDA_VISIBLE_DEVICES=$GPU

# generate

python test.py \
    --ckpt ${CKPT} \
    --test_set ${TEST_SET} \
    --save_dir ${SAVE_DIR} \
    --batch_size ${BATCH_SIZE} \
    --gpu 0 \
    --id ${ID}

#echo "Done generation"

