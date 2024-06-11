# Text Categorization Models

This repository contains various models for text categorization, including CNN, RNN, and several classical machine learning models optimized using Bayesian Optimization. The models are designed to work with text data, leveraging pre-trained embeddings for better performance.

## Table of Contents
- [Introduction](#introduction)
- [Setup](#setup)
- [Usage](#usage)
- [Models](#models)
- [Notebook](#notebook)

## Introduction

This project aims to provide implementations of several text categorization models, including:
- Convolutional Neural Networks (CNN)
- Recurrent Neural Networks (RNN)
- Logistic Regression (LR)
- Support Vector Classifier (SVC)
- Linear Support Vector Classifier (LinearSVC)
- Stochastic Gradient Descent (SGD)

Bayesian Optimization is used to fine-tune the hyperparameters of the classical machine learning models.

## Setup

To set up the environment and install the required dependencies, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your_username/text_categorization_models.git
    cd text_categorization_models
    ```

2. **Create a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The repository contains several Python scripts and a Jupyter notebook for training and evaluating the models. Below is a brief description of each component:

### Scripts

- **text_classification_model_optimization.py**: Contains functions to train and optimize LR, LinearSVC, and SVC models using Bayesian Optimization.
- **cnn_and_ensemble_model.py**: Defines the CNN model and ensemble methods for combining predictions from multiple models.
- **rnn_model.py**: Defines the RNN model.

### Jupyter Notebook

- **text_categorisation.ipynb**: Contains detailed steps for text preprocessing, model training, and evaluation. It also includes visualization of results and performance metrics.

### Running the Models

1. **Preprocess Data**: Use the Jupyter notebook to preprocess your text data and split it into training and validation sets.
2. **Train Models**: Train the desired model by running the corresponding script or by following the steps in the notebook.
3. **Evaluate Models**: Evaluate the trained models and visualize the results.

## Models

### Convolutional Neural Network (CNN)

The CNN model is defined in `cnn_and_ensemble_model.py`. It uses pre-trained embeddings and a series of convolutional layers to classify text data.

### Recurrent Neural Network (RNN)

The RNN model is defined in `rnn_model.py`. It employs LSTM layers to capture sequential dependencies in the text data.

### Classical Machine Learning Models

The scripts in `text_classification_model_optimization.py` provide implementations for Logistic Regression, SVC, and LinearSVC models with Bayesian Optimization for hyperparameter tuning.

## Notebook

The `text_categorisation.ipynb` notebook provides an end-to-end workflow for text categorization, including data preprocessing, model training, and evaluation.
