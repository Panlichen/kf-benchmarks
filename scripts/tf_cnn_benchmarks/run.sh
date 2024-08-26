#!/bin/bash
#SBATCH --gpus=1
module load anaconda3/2024.02-1 go/1.18.2 cmake/3.22.0 cuda/11.1 cudnn/8.1.1_cuda11.x gcc/7.5

source activate kungfu

# kungfu-run -np 2 python3 tf_cnn_benchmarks.py --num_gpus=1 --batch_size=32 --model=resnet50 --num_epochs=90 --variable_update=kungfu

# kungfu-run -np 2 python3 tf_cnn_benchmarks.py --num_gpus=1 --batch_size=32 --model=resnet50 --num_epochs=90 --variable_update=horovod

python tf_cnn_benchmarks.py --num_gpus=1 --batch_size=40 --model=resnet50 --variable_update=parameter_server
