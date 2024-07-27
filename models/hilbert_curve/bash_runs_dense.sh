#!/bin/bash

export KMP_WARNINGS=0

LOG_PREFIX="Hilbert_Curve_DenseNet"

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

run_batch_experiments() {
    local img_height=$1
    local img_width=$2
    local img_depth=$3
    local window_size=$4
    local hilbert_type=$5
    local epochs=$6
    local include_validation=$7

    SUBJECT=1
    run_experiment "$SUBJECT" "DENSE" "${LOG_PREFIX}_${BEST_CONFIG}" "$img_height" "$img_width" "$img_depth" "$window_size" 1 "$hilbert_type" "dense_params_${BEST_CONFIG}.json" 0.1 "$epochs" 1024 "$include_validation"
}

BEST_CONFIG=4
TIMESTAMP="$(date +"%Y%m%d%H%M%S")"

# Run experiments with different configurations
run_batch_experiments 16 10 1 16 "none" 60 ""
run_batch_experiments 4 4 10 16 "time" 60 ""
run_batch_experiments 4 4 16 16 "electrodes" 60 ""
run_batch_experiments 32 10 1 32 "none" 60 ""
run_batch_experiments 8 8 10 32 "time" 60 ""
run_batch_experiments 4 4 32 32 "electrodes" 60 ""
run_batch_experiments 64 10 1 64 "none" 60 ""
run_batch_experiments 8 8 10 64 "time" 60 ""
run_batch_experiments 4 4 64 64 "electrodes" 60 ""
