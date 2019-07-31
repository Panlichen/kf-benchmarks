#!/bin/bash

VERSION="1"
DATADIR="/fast/imagenet/train"

python tf_cnn_benchmarks.py \
	--data_format=NCHW \
	--batch_size=64 \
	--model=resnet50 \
	--optimizer=momentum \
	--variable_update=replicated \
	--nodistortions \
	--gradient_repacking=8 \
	--num_gpus=8 \
	--num_epochs=90 \
	--weight_decay=1e-4 \
	--data_dir=${DATADIR} \
	--checkpoint_interval=1 \
	--print_training_accuracy=True \
	>resnet-50-r${VERSION}.out 2>&1

echo "Bye."
exit 0
