# Implementing Transformer Models

## Overview
The intent of this project was to develop a deep understanding of the Transformer architecture and its implementation, as described in the paper 'Attention Is All You Need', using PyTorch.

## Table of Contents

- [Installation](#section-1)
- [Usage](#section-2)
- [Contributors](#section-3)

## [Installation](#section-1)
To get started with the project, follow these steps:

- Clone the repository to your local machine.
- Navigate to the project directory: ```cd ImplementingTransformerModels```
- Install dependencies: ```pip install -r requirements.txt```
  
## [Usage](#section-2)
Once installed, you can utilize the implemented Transformer models for translating english sentences to german. The main entry point is through the main.py script. All the settings can be configured in the ```settings.yaml``` file. Here, all available hyperparameters of the model can be specified, along with the name of the new training run and the logging levels. Additionally, the name of the pre-trained run can be specified for sampling or evaluation.

There are three use cases:
### Training
- ```python main.py --train```
- This command initiates a new training session with a model that has been randomly initialized.
- The program automatically creates a new folder in the ```run_logs``` directory. It creates the folder if it does not already exist.
- For each run, the program generates five plots (at 20% intervals of training progress) displaying the training and validation loss, which are then saved in the ```run_logs``` folder. 

### Sampling
- ```python main.py --sample```
- To specify the model to be used for sampling, set the name of the run in the variable ```SAMPLE_RUN_NAME``` in ```settings.py```.
- The model can now be used to translate a single English sentence into German.
- The user will be asked to enter an English sentence and will receive its German translation.

### Evaluating
- ```python main.py --bleu```
- This evaluates the model that is specified in the ```SAMPLE_RUN_NAME``` variable in the ```settings.py``` file.
- The script calculates and prints to the console the BLEU scores for all maximum orders from 1 to 4.