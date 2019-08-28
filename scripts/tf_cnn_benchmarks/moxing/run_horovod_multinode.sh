#!/usr/bin/env bash

set -x

cd /home/work/user-job-dir/tf_cnn_benchmarks/

chmod +x moxing/kube-plm-rsh-agent

KUBE_SA_CONFIG=/var/run/secrets/kubernetes.io/serviceaccount
if [ -d $KUBE_SA_CONFIG ]; then
    NAMESPACE=$(cat $KUBE_SA_CONFIG/namespace)
    TOKEN=$(cat $KUBE_SA_CONFIG/token)
fi

kubectl config set-cluster this --server https://kubernetes/ --certificate-authority=$KUBE_SA_CONFIG/ca.crt
kubectl config set-credentials me --token "$TOKEN"
kubectl config set-context me@this --cluster=this --user=me --namespace "$NAMESPACE"
kubectl config use me@this

kubectl get pods
kubectl config view


gen_hostfile() {
    pods=$(kubectl get pods -o name | grep job | awk -F '/' '{print $2}')
    for pod in $pods; do
        echo "$pod slots=8"
    done
}

gen_hostfile >moxing/hostfile


if [ "$DLS_TASK_INDEX" = "0" ]
then
mpirun --allow-run-as-root -np 16 \
    -mca plm_rsh_agent $PWD/moxing/kube-plm-rsh-agent \
    --hostfile $PWD/moxing/hostfile \
    --bind-to socket \
    -x LD_LIBRARY_PATH \
    -x NCCL_DEBUG=INFO -x NCCL_SOCKET_IFNAME=ib0,bond0,eth0 -x NCCL_SOCKET_FAMILY=AF_INET -x NCCL_IB_DISABLE=0 \
    -x HOROVOD_MPI_THREADS_DISABLE=1 \
    -mca pml ob1 -mca btl ^openib \
    -mca plm_rsh_no_tree_spawn true \
    -mca btl_tcp_if_include 192.168.0.0/16 \
    python tf_cnn_benchmarks.py --model=resnet50 \
    --data_name=imagenet \
    --num_batches=1000 \
    --eval=False \
    --forward_only=False \
    --print_training_accuracy=True \
    --num_gpus=1 \
    --num_warmup_batches=20 \
    --batch_size=64 \
    --momentum=0.9 \
    --weight_decay=0.0001 \
    --staged_vars=False \
    --variable_update=horovod \
    --optimizer=momentum \
    --use_datasets=True \
    --distortions=False \
    --fuse_decode_and_crop=True \
    --resize_method=bilinear \
    --display_every=50 \
    --checkpoint_every_n_epochs=False \
    --checkpoint_interval=0.25 \
    --checkpoint_directory=/cache/checkpoints-parallel \
    --data_format=NCHW \
    --batchnorm_persistent=True \
    --use_tf_layers=True \
    --winograd_nonfused=True
else
    sleep 5d
fi