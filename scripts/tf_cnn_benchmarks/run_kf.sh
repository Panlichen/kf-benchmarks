#!/bin/bash
#SBATCH --gpus=4
module load anaconda3/2024.02-1 go/1.18.2 cmake/3.22.0 cuda/11.1 cudnn/8.1.1_cuda11.x gcc/7.5

source activate kungfu

DATA_DIR=/HOME/scz1075/run/data/tiny-imagenet-200/tfrecords

# kungfu-run -np 2 python3 tf_cnn_benchmarks.py --num_gpus=1 --batch_size=32 --model=resnet50 --num_epochs=90 --variable_update=kungfu

# kungfu-run -np 2 python3 tf_cnn_benchmarks.py --num_gpus=1 --batch_size=32 --model=resnet50 --num_epochs=90 --variable_update=horovod

# python tf_cnn_benchmarks.py --num_gpus=1 --batch_size=40 --model=resnet50 --variable_update=parameter_server

# mpirun -np 2 python3 tf_cnn_benchmarks.py --batch_size=64 \
# --model=resnet50 --optimizer=momentum --variable_update=horovod \
# --nodistortions --num_gpus=1 \
# --num_epochs=90 --weight_decay=1e-4

# export PYTHONPATH=~/.local/lib/python3.9/site-packages:$PYTHONPATH
# kungfu-run -np 2 /HOME/scz1075/.conda/envs/kungfu/bin/python3 tf_cnn_benchmarks.py --batch_size=64 --data_format=NCHW \
# --model=resnet50 --optimizer=momentum --variable_update=kungfu --kungfu_option=sync_sgd \
# --nodistortions --num_gpus=1 \
# --num_epochs=90 --weight_decay=1e-4 --data_dir=${DATA_DIR}

python tf_cnn_benchmarks.py --num_gpus=4 --batch_size=40 --model=resnet50 --variable_update=parameter_server
