#!bin/bash/

for file in logs/*.json; do
    python3 metrics/metrics-all.py "${file}"
done