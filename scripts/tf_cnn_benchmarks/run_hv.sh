#!/bin/bash
#SBATCH --gpus=2
module load anaconda3/2024.02-1 go/1.18.2 cmake/3.22.0 cuda/11.1 cudnn/8.1.1_cuda11.x gcc/7.5
module load openmpi/4.1.1

source activate kungfu

# kungfu-run -np 2 python3 tf_cnn_benchmarks.py --num_gpus=1 --batch_size=32 --model=resnet50 --num_epochs=90 --variable_update=kungfu

# kungfu-run -np 2 python3 tf_cnn_benchmarks.py --num_gpus=1 --batch_size=32 --model=resnet50 --num_epochs=90 --variable_update=horovod

 #python tf_cnn_benchmarks.py --num_gpus=2 --batch_size=40 --model=resnet50 --variable_update=parameter_server

#python3 tf_cnn_benchmarks.py --batch_size=64 \
mpirun -np 2 python3 tf_cnn_benchmarks.py --data_format=NCHW --batch_size=64 \
--model=resnet50 --optimizer=momentum --variable_update=horovod \
--nodistortions --num_gpus=1 \
--num_epochs=90 --weight_decay=1e-4

# kungfu-run -np 2 python3 tf_cnn_benchmarks.py --batch_size=64 \
# --model=resnet50 --optimizer=momentum --variable_update=kungfu --kungfu_option=sync_sgd \
# --nodistortions --num_gpus=1 \
# --num_epochs=90 --weight_decay=1e-4

# python tf_cnn_benchmarks.py --data_format=NCHW --batch_size=256 \
# --model=resnet50 --optimizer=momentum --variable_update=replicated \
# --nodistortions --gradient_repacking=8 --num_gpus=2 \
# --num_epochs=90 --weight_decay=1e-4
