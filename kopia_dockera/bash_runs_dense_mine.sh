#!/bin/bash

LOG_PREFIX="Hilbert_Curve_DenseNet"
BEST_CONFIG=4

# Function to run the experiment
run_experiment() {
    local SUBJECTS=$1
    local EPOCHS=$2
    local IMG_HEIGHT=$3
    local IMG_WIDTH=$4
    local IMG_DEPTH=$5
    local HILBERT_TYPE=$6
    local WINDOW_SIZE=$7

    TIMESTAMP="$(date +"%Y%m%d%H%M%S")"
    for SUBJECT in $SUBJECTS
    do
        python3 run_experiment_hilbert.py --subject $SUBJECT --model "DENSE" \
        --timestamp $TIMESTAMP --log "${LOG_PREFIX}_Model_${BEST_CONFIG}" \
        --include_rest_gesture --img_height $IMG_HEIGHT --img_width $IMG_WIDTH --img_depth $IMG_DEPTH \
        --window_size $WINDOW_SIZE --window_step 1 \
        --augment_factor 1 \
        --augment_jitter 25 --augment_mwrp 0.2 \
        --hilbert_type $HILBERT_TYPE \
        --model_params "dense_params_${BEST_CONFIG}.json" --dropout 0.1 \
        --epochs $EPOCHS --batch_size 1024
    done
}

# Define subjects and run experiments with various configurations
SUBJECTS=$(seq 1 2)
#run_experiment "$SUBJECTS" 3 16 10 1 "none" 16
#run_experiment "$SUBJECTS" 60 4 4 10 "time" 16
run_experiment "$SUBJECTS" 60 4 4 16 "electrodes" 16
#run_experiment "$SUBJECTS" 60 32 10 1 "none" 32
#run_experiment "$SUBJECTS" 60 8 8 10 "time" 32
#run_experiment "$SUBJECTS" 60 4 4 32 "electrodes" 32
#run_experiment "$SUBJECTS" 60 64 10 1 "none" 64
#run_experiment "$SUBJECTS" 60 8 8 10 "time" 64
#run_experiment "$SUBJECTS" 60 4 4 64 "electrodes" 64
