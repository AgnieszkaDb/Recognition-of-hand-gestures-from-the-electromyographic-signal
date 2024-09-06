#!/bin/bash

center() {
  termwidth="$(tput cols)"
  padding="$(printf '%0.1s' ={1..500})"
  printf '\n%*.*s %s %*.*s\n\n' 0 "$(((termwidth-2-${#1})/2))" "$padding" "$1" 0 "$(((termwidth-1-${#1})/2))" "$padding"
}

mkdir -p logs
mkdir -p metrics
declare -A models

models[ATTENTION-MODEL]="models/Attention-model,th25,python3,run.py"
models[BASELINE]="models/Baseline/src,tf2.10,python3,run.py"
models[HILBERT]="models/Hilbert,tf115,bash,bash_runs_mshilb.sh"
models[DUAL-STREAM-LSTM]="models/LSTM-CNN-Inception,tf2.10,python3,run.py"
models[MCMP-NET]="models/MCMP-Net,tf2.10,python3,run.py"

for model in "${!models[@]}"; do
    IFS=',' read -r path env_name exec script <<< "${models[$model]}"
    center "RUNNING $model"

    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate "$env_name"

    cd "$path" || exit
    if [ "$model" = "HILBERT" ]; then
      python3 Datasets/create_dataset_db1.py
    fi

    "$exec" "$script"

    conda deactivate
    cd ../..
done