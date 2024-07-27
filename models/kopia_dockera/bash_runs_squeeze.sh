#!/bin/bash

LOG_PREFIX="Hilbert_Curve_SqueezeNet"
TIMESTAMP="$(date +"%Y%m%d%H%M%S")"
BEST_CONFIG=1
export KMP_WARNINGS=0


run_experiment() {
    local subject=$1
    local model=$2
    local log_prefix=$3
    local img_height=$4
    local img_width=$5
    local img_depth=$6
    local window_size=$7
    local window_step=$8
    local hilbert_type=$9
    local model_params=${10}
    local dropout=${11}
    local epochs=${12}
    local batch_size=${13}
    local include_validation=${14}

    python3 run_experiment_hilbert.py --subject "$subject" --model "$model" \
        --timestamp "$TIMESTAMP" --log "$log_prefix" \
        --include_rest_gesture --img_height "$img_height" --img_width "$img_width" --img_depth "$img_depth" \
        --window_size "$window_size" --window_step "$window_step" \
        --augment_jitter 25 --augment_mwrp 0.2 \
        --hilbert_type "$hilbert_type" \
        --model_params "$model_params" --dropout "$dropout" \
        --epochs "$epochs" --batch_size "$batch_size" \
        $include_validation
}

# SQUEEZE Exps window 16
for i in {0..5..1}; do
    for SUBJECT in 2 21 6 8 5 10 18 16 13 14; do
        run_experiment "$SUBJECT" "SQUEEZE" "${LOG_PREFIX}_Model_$i" 16 10 1 16 1 "none" "squeeze_params_$i.json" 0.1 30 1024 "--validation"
    done
done

# Different configurations
CONFIGURATIONS=(
    "4 4 10 16 1 time"
    "4 4 16 16 1 electrodes"
    "32 10 1 32 1 none"
    "8 8 10 32 1 time"
    "4 4 32 32 1 electrodes"
    "64 10 1 64 1 none"
    "8 8 10 64 1 time"
    "4 4 64 64 1 electrodes"
)

for config in "${CONFIGURATIONS[@]}"; do
    IMG_HEIGHT=$(echo $config | awk '{print $1}')
    IMG_WIDTH=$(echo $config | awk '{print $2}')
    IMG_DEPTH=$(echo $config | awk '{print $3}')
    WINDOW_SIZE=$(echo $config | awk '{print $4}')
    WINDOW_STEP=$(echo $config | awk '{print $5}')
    HILBERT_TYPE=$(echo $config | awk '{print $6}')
    
    SUBJECT=1
    while [ $SUBJECT -lt 28 ]; do
        run_experiment "$SUBJECT" "SQUEEZE" "${LOG_PREFIX}_${BEST_CONFIG}" "$IMG_HEIGHT" "$IMG_WIDTH" "$IMG_DEPTH" "$WINDOW_SIZE" "$WINDOW_STEP" "$HILBERT_TYPE" "squeeze_params_${BEST_CONFIG}.json" 0.1 60 1024 ""
        SUBJECT=$((SUBJECT + 1))
    done
done
