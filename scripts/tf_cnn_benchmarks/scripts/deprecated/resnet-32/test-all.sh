


for parts in {1..14}
do
   for worker in 0 1 2 3
   do
        CHECKPOINT_DIRECTORY="/data/kungfu/checkpoints/andrei-checkpoints/kungfu-logs-validation/checkpoints-ako-${parts}-worker-${worker}/v-000001"
        ./scripts/test-resnet-32.sh ${CHECKPOINT_DIRECTORY} > "/data/kungfu/checkpoints/andrei-checkpoints/exhaustive-validation/validation-ako-${parts}-worker-${worker}" 2>&1
   done
done

for worker in 0 1 2 3
do
    CHECKPOINT_DIRECTORY="/data/kungfu/checkpoints/andrei-checkpoints/kungfu-logs-validation/checkpoints-parallel-worker-${worker}/v-000001"
    ./scripts/test-resnet-32.sh ${CHECKPOINT_DIRECTORY} > "/data/kungfu/checkpoints/andrei-checkpoints/exhaustive-validation/validation-parallel-worker-${worker}" 2>&1
done