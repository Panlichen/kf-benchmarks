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

# curl -v --cacert $KUBE_SA_CONFIG/ca.crt https://kubernetes/ -H "Authorization: Bearer $TOKEN"
# kubectl get ns
kubectl get pods
# kubectl get deployments
# kubectl get ds
# kubectl get sa
# kubectl get cm


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