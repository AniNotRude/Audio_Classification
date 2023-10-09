# Urban Sound Classification with PyTorch

This Python script is designed to train a Convolutional Neural Network (CNN) for classifying urban sounds using the UrbanSound8K dataset(https://urbansounddataset.weebly.com/urbansound8k.html). The script follows a step-by-step process for data loading, preprocessing, model definition, training, and inference.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Usage](#usage)

## Overview

Urban sound classification is a common task in audio analysis, and this script provides a practical example using PyTorch. It covers the following key steps:

1. Imports the necessary libraries and modules.
2. Defines a class mapping for sound categories.
3. Implements a `predict` function for model evaluation.
4. Sets constants and file paths.
5. Creates a data loader for training data.
6. Defines functions for training a single epoch and the entire model.
7. Implements a custom dataset class for audio data loading and preprocessing.
8. Defines a CNN model architecture.
9. Trains the model, saves its state, and performs inference.

## Requirements

To run this script, you need the following prerequisites:

- Python 3.x
- PyTorch
- torchaudio
- pandas
- torchsummary

You can install these dependencies using `pip`:
```
pip install torch torchaudio pandas torchsummary
```

## Usage
```
git clone https://github.com/AniNotRude/Audio_Classification.git
```



