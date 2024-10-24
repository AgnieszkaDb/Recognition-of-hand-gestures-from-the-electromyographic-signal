' # RECOGNITION OF HAND GESTURES FROM THE ELECTROMYOGRAPHIC SIGNAL

This repository contains the code and resources used in the master's thesis titled **"RECOGNITION OF HAND GESTURES FROM THE ELECTROMYOGRAPHIC SIGNAL"**.

## Overview

The project focuses on recognizing hand gestures using electromyographic (EMG) signals by comparing the performance of five different models:

1. **Baseline Model**: Implemented by the author as part of a bachelor's thesis.
2. **Hilbert Data Scanning CNN**: A convolutional neural network (CNN) utilizing Hilbert transform for data scanning.
3. **MCMP-Net**: A specialized architecture designed for EMG signal recognition.
4. **Dual Stream LSTM Classifier**: Long Short-Term Memory (LSTM)-based model with dual data streams for improved classification.
5. **Attention Model**: A neural network model leveraging attention mechanisms for enhanced performance.

This project aims to explore and compare the effectiveness of these models in recognizing hand gestures from EMG signals.

## Requirements

To ensure compatibility, this project requires **CUDA compilation tools version: 11.5** for GPU acceleration.

## Setting Up the Environment

To set up the necessary conda environments, navigate to the `conda` directory and run the following script:

```bash
cd conda
bash load_envs.sh
```

## Running Experiments

To run all the experiments, execute the following command:

```bash
bash run_all.bash
```

## Generating Metrics

After running the experiments, you can generate the evaluation metrics for model comparison by running:

```bash
bash metrics.bash
```

