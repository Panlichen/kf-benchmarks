#!/usr/bin/env bash

NUM_WORKERS=4

train() {
    BATCH=$1
    TRAIN_ID=$2
    echo "[BEGIN TRAINING KEY] training-parallel-${TRAIN_ID}"
    kungfu-prun  -np ${NUM_WORKERS} -H 127.0.0.1:${NUM_WORKERS} -timeout 1000000s \
        python3 tf_cnn_benchmarks.py --model=resnet32 --data_name=cifar10 --data_dir=/data/cifar-10/cifar-10-batches-py \
        --num_epochs=1 \
        --num_gpus=1 \
        --eval=False \
        --forward_only=False \
        --num_warmup_batches=20 \
        --print_training_accuracy=True \
        --batch_size=${BATCH} \
        --momentum=0.9 \
        --weight_decay=0.0001 \
        --optimizer=momentum \
        --staged_vars=False \
        --restore_checkpoint=False \
        --variable_update=kungfu \
        --kungfu_strategy=cpu_all_reduce_noise \
        --use_datasets=True \
        --distortions=False \
        --fuse_decode_and_crop=True \
        --resize_method=bilinear \
        --display_every=1 \
        --checkpoint_directory=/data/kungfu/train_dir/checkpoint \
        --checkpoint_every_n_epochs=True \
        --checkpoint_interval=0.25 \
        --data_format=NCHW \
        --batchnorm_persistent=True \
        --use_tf_layers=True \
        --winograd_nonfused=True 
    echo "[END TRAINING KEY] training-parallel-${TRAIN_ID}"
}

validate() {
    VALIDATION_ID=$1
    for worker in 0 # 1 2 3 
    do
    echo "[BEGIN VALIDATION KEY] validation-parallel-worker-${worker}-validation-id-${VALIDATION_ID}"
    python3 tf_cnn_benchmarks.py --eval=True --forward_only=False --model=resnet32 --data_name=cifar10 \
        --data_dir=/data/cifar-10/cifar-10-batches-py \
        --variable_update=replicated --data_format=NCHW --use_datasets=False --num_batches=50 --eval_batch_size=150 \
        --num_gpus=4 --use_tf_layers=True \
        --checkpoint_interval=0.25 \
        --checkpoint_directory=/data/kungfu/train_dir/checkpoint-worker-${worker}/v-000001  \
        --checkpoint_every_n_epochs=False 
    echo "[END VALIDATION KEY] validation-parallel-worker-${worker}-validation-id-${VALIDATION_ID}"
    done
}


NUM_EPOCHS="1"
NOISE_FILES_PATH="/home/ab7515"
NEW_NOISE_FILE_NAME="${NOISE_FILES_PATH}/median-noise.txt"

i="0"

FUTURE_BATCH=32

# Train
start=`date +%s`
train ${FUTURE_BATCH} ${i} ${NUM_EPOCHS}
end=`date +%s`
runtime_train=$((end-start))
echo "Train ${i} took: ${runtime_train}"


# Validate
start=`date +%s`
validate ${i}
end=`date +%s`
runtime_val=$((end-start))
echo "Validation ${i} took: ${runtime_val}"


