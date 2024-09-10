#!/bin/bash

declare -a yaml_files=("env1.yml" "env2.yml" "env3.yml")

for yaml_file in "${yaml_files[@]}"; do
    env_name=$(grep '^name:' "$yaml_file" | awk '{print $2}')
    
    if [ -z "$env_name" ]; then
        echo "No environment name found in $yaml_file"
        continue
    fi
    
    echo "Creating environment $env_name from $yaml_file"
    conda env create -f "$yaml_file"
    
    if [ $? -eq 0 ]; then
        echo "Successfully created environment $env_name"
    else
        echo "Failed to create environment $env_name"
    fi
done

