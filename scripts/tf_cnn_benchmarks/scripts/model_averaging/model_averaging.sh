#!/usr/bin/env bash

RUN=1

mkdir /home/ab7515/checkpoints-test

train() {
    BATCH=$1
    echo "[BEGIN TRAINING KEY] training-lbr-${RUN}"
    # --data_dir=/data/imagenet/records
    kungfu-prun  -np 4 -H 127.0.0.1:4 -timeout 1000000s \
        python3 tf_cnn_benchmarks.py --model=resnet32 --data_name=cifar10  \
        --num_epochs=5 \
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
        --optimizer=adaptive_model_averaging \
        --kungfu_strategy=adaptive \
        --model_averaging_device=gpu \
        --request_mode=sync \
        --use_datasets=True \
        --distortions=False \
        --fuse_decode_and_crop=True \
        --resize_method=bilinear \
        --display_every=100 \
        --checkpoint_every_n_epochs=False \
        --checkpoint_interval=0.25 \
        --checkpoint_directory=/home/ab7515/checkpoints-test \
        --data_format=NHWC \
        --batchnorm_persistent=True \
        --use_tf_layers=True \
        --winograd_nonfused=True
    echo "[END TRAINING KEY] training-lbr-${RUN}"
}

validate() {
    for worker in 0 # 1 2 3 4 5 6 7
    do
    echo "[BEGIN VALIDATION KEY] validation-lbr-${RUN}-worker-${worker}"
    python3 tf_cnn_benchmarks.py --eval=True --forward_only=False --model=resnet32 --data_name=cifar10 \
        --data_dir=/data/cifar-10/cifar-10-batches-py \
        --variable_update=replicated --data_format=NCHW --use_datasets=False --num_batches=50 --eval_batch_size=50 \
        --num_gpus=4 --use_tf_layers=True \
        --checkpoint_directory=/data/kungfu/checkpoints-lbr/checkpoint-worker-${worker}/v-000001 --checkpoint_interval=0.25 \
        --checkpoint_every_n_epochs=True 
    echo "[END VALIDATION KEY] validation-lbr-${RUN}-worker-${worker}"
    done
}


train 64
#validate