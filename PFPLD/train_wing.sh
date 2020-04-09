#!/usr/bin/env bash
Loss=wing
MODELDIR="./models/checkpoint/model-$Loss"
mkdir -p "$MODELDIR"
LOGFILE="$MODELDIR/log"

CUDA_VISIBLE_DEVICES='0' python -u train.py\
  --dataroot "/home/unaguo/hanson/data/landmark/WFLW191104/train_data/list.txt" \
  --val_dataroot "/home/unaguo/hanson/data/landmark/WFLW191104/test_data/list.txt" \
  --snapshot "$MODELDIR" \
  --tensorboard "$MODELDIR/tensorboard" \
  --resume "./models/checkpoint/model-wing/checkpoint_epoch_126.pth" \
  --loss "$Loss" \
  --workers 8 \
  --base_lr 1e-2 \
  --train_batchsize 192 \
  --val_batchsize 8 \
  > "$LOGFILE" 2>&1 &