#!/bin/bash

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

cd /#######################
torchrun --nproc_per_node=2 --nnode=1 --master_port=${PORT} trainer.py \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LR} \
    --epochs ${EPOCHS} \
    --alpha ${ALPHA}

    #--num_bins ${NUM_BINS}
    # --device ${DEVICE}  

