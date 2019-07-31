#!/bin/bash
#
# Benchmark(s) for TensorFlow.
#
PROFILE=0

NVPROFEXEC=nvprof
NVPROFOPTS="-o profiler.nvvp"
if [ $PROFILE -gt 1 ]; then
	#
	# Detailed profiling looks into SM occupancy and activity.
	# Run only 1 batch on 1 GPU.
	#
	NVPROFOPTS=${NVPROFOPTS}" --metrics achieved_occupancy,sm_activity"
fi

# Batch size (per GPU)
B="32"
# Numbe of GPU devices
G="4"
# FIXME: divide learning rate by the number of workers
# Learning rate schedule (step-wise)
S="0.1;80;0.01;120;0.001" # LR0;E1;LR1;...;En;LRn
# Version of ResNet
V="0"

# Read from the command line...
# B=$1
# G=$2
# S=$3


FLAGS=

#
# Basic configuration
#

MODEL="resnet50"

DATASET="imagenet"

# Synthetic data
# DATADIR=""
# TF record data
DATADIR="/data/imagenet/records"
# DATADIR="datasets/cifar-10"

EPOCHS=7 # 160

MOMENTUM=0.9
DECAY="0.0001"
OPTIMISER="momentum" # "sgd" or "momentum"

# Synchronisation parameters

UPDATE="replicated" # parameter_server, replicated, independent, kungfu
STAGED="False"
ALLREDUCE="" # empty, nccl, or xring


# KungFu Synchronization strategy
KUNGFU_STRATEGY="" # ako, cpu_all_reduce, nccl_all_reduce
AKO_PARTITIONS=""   # empty, max_number_of_model_variables

DISPLAY_INTERVAL=100

# Also checkpoint every N epochs
# CHECKPOINT_EVERY_N_EPOCHS=True

# We want 1 checkpoint per epoch...
CHECKPOINT_INTERVAL=1
CHECKPOINT_DIRECTORY="/data/kungfu/checkpoints/andrei-checkpoints/resnet-50/kungfu-logs-validation/checkpoints-replicated-correct"

FLAGS=${FLAGS}" --model=${MODEL}"
FLAGS=${FLAGS}" --data_name=${DATASET}"

[ -n "${DATADIR}" ] && FLAGS=${FLAGS}" --data_dir=${DATADIR}"

#if [ $PROFILE -gt 1 ]; then
#	# When profiling SM activity,
#	# run only 1 batch.
#	FLAGS=${FLAGS}" --num_batches=1"
#else
#	FLAGS=${FLAGS}" --num_epochs=${EPOCHS}"
#fi

FLAGS=${FLAGS}" --num_epochs=${EPOCHS}"
#FLAGS=${FLAGS}" --num_batches=1"

# Run benchmark in training mode
FLAGS=${FLAGS}" --eval=False --forward_only=False"

# Print training accuracy
FLAGS=${FLAGS}" --print_training_accuracy=True" 

# Set default random seed
FLAGS=${FLAGS}" --tf_random_seed=123456789"

# CPU/GPU configuration
if [ $PROFILE -gt 1 ]; then
	# Overide configuration
	G=1
fi
FLAGS=${FLAGS}" --num_gpus=$G --gpu_thread_mode=gpu_private"

# No warm-up
FLAGS=${FLAGS}" --num_warmup_batches=20"

# Hyper-parameters
FLAGS=${FLAGS}" --batch_size=${B}"

# By default learning_rate is None, in which case we fall 
# back to a model-specific learning rate
# 
# By default `--piecewise_learning_rate_schedule` is None
[ -n "${S}" ] && FLAGS=${FLAGS}" --piecewise_learning_rate_schedule=${S}"

FLAGS=${FLAGS}" --momentum=${MOMENTUM}"
FLAGS=${FLAGS}" --weight_decay=${DECAY}"
FLAGS=${FLAGS}" --optimizer=${OPTIMISER}"

# Synchronisation parameters
FLAGS=${FLAGS}" --variable_update=${UPDATE}"
FLAGS=${FLAGS}" --staged_vars=${STAGED}"
[ -n "${ALLREDUCE}" ] && FLAGS=${FLAGS}" --all_reduce_spec=${ALLREDUCE}"

# Synchronization strategy
FLAGS=${FLAGS}" --kungfu_strategy=${KUNGFU_STRATEGY}"
[ -n "${AKO_PARTITIONS}" ] && FLAGS=${FLAGS}" --ako_partitions=${AKO_PARTITIONS}"

# Image producer configuration
FLAGS=${FLAGS}" --use_datasets=True"

# Image pre-processor configuration
FLAGS=${FLAGS}" --distortions=False"
FLAGS=${FLAGS}" --fuse_decode_and_crop=True"
FLAGS=${FLAGS}" --resize_method=bilinear"

# Checkpoints
#
# Fix display interval to every 100 tasks...
FLAGS=${FLAGS}" --display_every=${DISPLAY_INTERVAL}"

FLAGS=${FLAGS}" --checkpoint_every_n_epochs=True"
FLAGS=${FLAGS}" --checkpoint_interval=${CHECKPOINT_INTERVAL}"
FLAGS=${FLAGS}" --checkpoint_directory=${CHECKPOINT_DIRECTORY}"

FLAGS=${FLAGS}" --data_format=NCHW"
FLAGS=${FLAGS}" --batchnorm_persistent=True"
FLAGS=${FLAGS}" --use_tf_layers=True"
FLAGS=${FLAGS}" --winograd_nonfused=True"


# Run
printf "Train ResNet32 v.%s with batch size %4d on %d GPU devices\n" $V $B $G
echo "python3 tf_cnn_benchmarks.py $FLAGS"
python3 tf_cnn_benchmarks.py $FLAGS > /data/kungfu/checkpoints/andrei-checkpoints/resnet-50/resnet-50-b-$B-g-$G-replicated-correct.out 2>&1


# Validation
./scripts/resnet-50/test-resnet-50.sh ${CHECKPOINT_DIRECTORY} > "/data/kungfu/checkpoints/andrei-checkpoints/resnet-50/kungfu-logs-validation/validation-replicated" 2>&1
