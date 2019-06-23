#!/usr/bin/env bash


mkdir /data/kungfu/checkpoints-lbr

######## pkill -f python3
# cd /ab7515/KungFu
# env KUNGFU_USE_NCCL=0 pip3 install --no-index --user -U .
# cd /ab7515/benchmarks/benchmarks-fresh/scripts/tf_cnn_benchmarks



# cd /home/ab7515/KungFu
# env KUNGFU_USE_NCCL=1 pip3 install --no-index .
# cd /home/ab7515/benchmarks/benchmarks-fresh/scripts/tf_cnn_benchmarks


# export KUNGFU_CONFIG_ENABLE_MONITORING=true
# export KUNGFU_CONFIG_MONITORING_PERIOD=10s


# --model=resnet50 --data_name=imagenet --data_dir=/data/imagenet/records

RUN=1
#adaptive_partial_exchange_with_cpu_allreduce

# export KUNGFU_CONFIG_ENABLE_LATENCY_MONITORING=true

mkdir /home/ab7515/abc

train() {
    BATCH=$1
    echo "[BEGIN TRAINING KEY] training-lbr-${RUN}"
    N_PEERS=4
    # --data_dir=/data/imagenet/records
    kungfu-prun  -np ${N_PEERS} -H 127.0.0.1:${N_PEERS} -timeout 1000000s \
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
        --optimizer=p2p_averaging \
        --variable_update=kungfu \
        --kungfu_strategy=none \
        --model_averaging_device=gpu \
        --request_mode=sync \
        --window_size=10 \
        --use_datasets=True \
        --distortions=False \
        --fuse_decode_and_crop=True \
        --resize_method=bilinear \
        --display_every=100 \
        --checkpoint_every_n_epochs=False \
        --checkpoint_interval=0.25 \
        --checkpoint_directory=/home/ab7515/abc \
        --data_format=NHWC \
        --batchnorm_persistent=True \
        --use_tf_layers=True \
        --winograd_nonfused=True
    echo "[END TRAINING KEY] training-lbr-${RUN}"
}

validate() {
    for worker in 0 #  2 3 # 4 5 6 7
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