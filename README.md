# Sentiment Classification Using RNNs

This project implements a Sentiment Classification model using Recurrent Neural Networks (RNNs) to predict whether a given IMDB movie review is positive or negative. The project includes data preprocessing, model building, training, and evaluation. Additionally, hyperparameter tuning and experiments with LSTM layers are conducted.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Tasks](#tasks)
- [Implementation](#implementation)
- [Results](#results)
- [Usage](#usage)
- [References](#references)

## Introduction

Sentiment classification is a common task in Natural Language Processing (NLP) where the goal is to determine the sentiment of a given text. In this project, we use RNNs to classify IMDB movie reviews as positive or negative.

## Dataset

The dataset used is the IMDB Movie Review Dataset, which contains 50,000 movie reviews labeled as positive or negative.

- [IMDB Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)

## Tasks

### 1. RNN Model

#### 1.1 Build a Dataset

- Create a dataset from the IMDB Movie Review Dataset by filtering reviews with word counts between 100 and 500.
- Perform text processing on the reviews and create a word-to-index mapping to represent any review as a list of numbers.

#### 1.2 Create Dataloaders

- Create dataloaders for the training, test, and validation datasets with appropriate batch sizes.

#### 1.3 Create the Model

- Implement the Model class for the RNN model.
- Create functions for training and testing the model.
- Experiment with only picking the last output and the mean of all outputs in the RNN layer and observe the changes.

### 2. Hyperparameter Tuning

- Starting with the best configurations based on the above experiments, experiment with 5 different hyperparameter configurations. 
- You can change the size of the embedding layer, hidden state, and batch size in the dataloader.

### 3. After RNNs

- Keeping all parameters the same, replace the RNN layer with the LSTM layer using `nn.LSTM`.
- Observe the changes and explain why the LSTM layer would affect performance.

## Implementation

### 1. Data Preparation

- **Text Processing**: Clean the text data by removing special characters and performing tokenization.
- **Word-to-Index Mapping**: Create a vocabulary and map each word to a unique index.
- **Dataset Filtering**: Filter reviews based on word count (between 100 and 500).

### 2. Model Building

- **RNN Model**: Implement the RNN model class with options to pick the last output or the mean of all outputs.
- **LSTM Model**: Implement the LSTM model class by replacing the RNN layer with `nn.LSTM`.

### 3. Training and Evaluation

- **Training Loop**: Implement the training loop with logging of training and validation losses and accuracies using wandb.
- **Evaluation**: Evaluate the model on the test set and compute the accuracy.

### 4. Hyperparameter Tuning

- Experiment with different hyperparameters and record the results.

## Results

- **Loss and Accuracy Graphs**: Plot graphs for loss and accuracy for each epoch of the training loop.
- **Hyperparameter Tuning Results**: Record observations for different hyperparameter configurations.
- **RNN vs LSTM**: Compare the performance of the RNN and LSTM models and explain the differences.

## Usage

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/NithinV-tech/SENTIMENT-CLASSIFICATION-RNN.git
   cd SENTIMENT-CLASSIFICATION-RNN
