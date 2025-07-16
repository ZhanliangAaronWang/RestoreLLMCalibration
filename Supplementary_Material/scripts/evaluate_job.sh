#!/bin/bash

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False


cd ##################
python inference_temperature.py \
    --batch_size ${BATCH_SIZE} \
    --alpha ${ALPHA} \
    #--num_bins ${NUM_BINS}
    # --device ${DEVICE}  
