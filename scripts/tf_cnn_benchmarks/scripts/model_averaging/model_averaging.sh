#!/usr/bin/env bash

RUN=1

mkdir /data/kungfu/checkpoints-test

train() {
    BATCH=$1
    echo "[BEGIN TRAINING KEY] training-${RUN}"
    kungfu-prun  -np 4 -H 127.0.0.1:4 -timeout 1000000s \
        python3 tf_cnn_benchmarks.py \
        --model=resnet50 \
        --data_name=imagenet \
        --data_dir=/data/imagenet/records/ \
        --num_batches=500 \
        --eval=False \
        --forward_only=False \
        --print_training_accuracy=True \
        --num_gpus=1 \
        --num_warmup_batches=20 \
        --batch_size=${BATCH} \
        --momentum=0.9 \
        --weight_decay=0.0001 \
        --staged_vars=False \
        --variable_update=kungfu \
        --optimizer=model_averaging \
        --kungfu_strategy=monitoring_static \
        --model_averaging_device=gpu \
        --request_mode=sync \
        --use_datasets=True \
        --distortions=False \
        --fuse_decode_and_crop=True \
        --resize_method=bilinear \
        --display_every=100 \
        --checkpoint_every_n_epochs=True \
        --checkpoint_interval=0.25 \
        --checkpoint_directory=/data/kungfu/checkpoints-test/checkpoint-test \
        --data_format=NHWC \
        --batchnorm_persistent=True \
        --use_tf_layers=True \
        --winograd_nonfused=True
    echo "[END TRAINING KEY] training-${RUN}"
}

validate() {
    for worker in 0 1 2 3
    do
    echo "[BEGIN VALIDATION KEY] validation-${RUN}-worker-${worker}"
    python3 tf_cnn_benchmarks.py \
        --eval=True \
        --forward_only=False \
        --model=resnet50 \
        --data_name=imagenet \
        --data_dir=/data/imagenet/records/ \
        --variable_update=replicated --data_format=NCHW --use_datasets=False --num_batches=50 --eval_batch_size=50 \
        --num_gpus=4 --use_tf_layers=True \
        --checkpoint_directory=/data/kungfu/checkpoints-test/checkpoint-test-worker-${worker}/v-000001 --checkpoint_interval=0.25 \
        --checkpoint_every_n_epochs=True 
    echo "[END VALIDATION KEY] validation-${RUN}-worker-${worker}"
    done
}


train 64
#validate