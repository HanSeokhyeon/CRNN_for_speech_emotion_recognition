#!/bin/sh

export CUDA_VISIBLE_DEVICES="0"

BATCH_SIZE=8
WORKER_SIZE=4
MAX_EPOCHS=100

python ./main.py --batch_size $BATCH_SIZE --workers $WORKER_SIZE --bidirectional --max_epochs $MAX_EPOCHS
