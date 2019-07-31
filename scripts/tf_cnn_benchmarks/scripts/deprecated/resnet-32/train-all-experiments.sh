KUNGFU_STRATEGY_1="partial_exchange"

KUNGFU_STRATEGY_2="partial_exchange_accumulation"

KUNGFU_STRATEGY_3="partial_exchange_accumulation_avg_peers"

KUNGFU_STRATEGY_4="partial_exchange_accumulation_avg_window"


for KUNGFU_STRATEGY in KUNGFU_STRATEGY_1 # KUNGFU_STRATEGY_2 KUNGFU_STRATEGY_3 KUNGFU_STRATEGY_4
do
    for fraction in 0.1 # 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
    do
        ./scripts/train-resnet-32-partial-exchange.sh ${fraction} ${KUNGFU_STRATEGY}
    done
    # for fraction in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
    # do
    #     for worker in 0 1 2 3
    #     do
    #         CHECKPOINT_DIRECTORY="/data/kungfu/checkpoints/andrei-checkpoints/kungfu-logs-${KUNGFU_STRATEGY}/checkpoints-${KUNGFU_STRATEGY}-fraction-${fraction}-worker-${worker}/v-000001"
    #         ./scripts/test-resnet-32.sh ${CHECKPOINT_DIRECTORY} > "/data/kungfu/checkpoints/andrei-checkpoints/kungfu-logs-${KUNGFU_STRATEGY}/validation-${KUNGFU_STRATEGY}-fraction-${fraction}-worker-${worker}" 2>&1
    #     done
    # done
done

