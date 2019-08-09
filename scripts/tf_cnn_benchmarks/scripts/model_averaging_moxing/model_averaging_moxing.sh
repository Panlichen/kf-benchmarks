#!/usr/bin/env bash

set -x

cd /home/work/user-job-dir/benchmarks-paper/scripts/tf_cnn_benchmarks/

BATCH_SIZE=64

kungfu-prun -np 16 -H 169.254.128.207:8,169.254.128.185:8 -nic ib0 -timeout 1000000s \
    python tf_cnn_benchmarks.py --model=resnet50 \
    --data_name=imagenet \
    --num_batches=1000 \
    --eval=False \
    --forward_only=False \
    --print_training_accuracy=True \
    --num_gpus=1 \
    --num_warmup_batches=20 \
    --batch_size=${BATCH_SIZE} \
    --momentum=0.9 \
    --weight_decay=0.0001 \
    --staged_vars=False \
    --variable_update=kungfu \
    --optimizer=adaptive_model_averaging \
    --kungfu_strategy=adaptive \
    --model_averaging_device=gpu \
    --request_mode=sync \
    --use_datasets=True \
    --distortions=False \
    --fuse_decode_and_crop=True \
    --resize_method=bilinear \
    --display_every=50 \
    --checkpoint_every_n_epochs=False \
    --checkpoint_interval=0.125 \
    --checkpoint_directory=/cache/checkpoints-parallel \
    --data_format=NCHW \
    --batchnorm_persistent=True \
    --use_tf_layers=True \
    --winograd_nonfused=True 
