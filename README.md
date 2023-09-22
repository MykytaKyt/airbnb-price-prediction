# House Price Prediction Project

This project focuses on predicting house prices using machine learning models. It encompasses various stages, including data processing, text processing, feature engineering, outlier detection, model training, and evaluation.

## Table of Contents

1. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Getting the Data](#getting-the-data)

2. [Installation](#installation)

3. [Usage](#usage)
   - [Data Processing](#data-processing)
   - [Text Processing](#text-processing)
   - [Feature Engineering](#feature-engineering)
   - [Outlier Detection](#outlier-detection)
   - [Model Training](#model-training)

4. [Results](#results)

## Getting Started

### Prerequisites

Before you begin, make sure you have the following prerequisites installed:

- **Python 3.x:** This project is written in Python, so you'll need a Python 3.x environment.
- **pip (Python package manager):** You can use `pip` to easily install the required libraries and dependencies.

### Getting the Data

To obtain the dataset for this project, follow these steps:

1. Visit the [Inside Airbnb](http://insideairbnb.com/get-the-data/) website.
2. Download the dataset archive that matches your project's requirements.
3. Extract the downloaded archive to the project directory.

Alternatively, you can use the provided `train.zip` file; simply unzip it to access the data.

## Installation

To install the necessary Python libraries and dependencies, execute the following command:

```bash
pip install -r requirements.txt
```

## Usage

This project utilizes a `Makefile` to streamline common tasks. Here's how to use it:

To run the entire pipeline from data processing to model training, execute:

```bash
make all
```

### Data Processing

Data preprocessing is the first step in the pipeline. It involves cleaning and preparing the dataset for analysis. The `data_processing.py` script performs the following tasks:

- Loads the dataset from the `data/train.csv` file.
- Cleans the dataset by removing unnecessary columns and converting some columns to appropriate data types.
- Performs data transformations like currency conversion and percentage formatting.
- Saves the preprocessed data to `data/prep.csv`.

To preprocess the data, run:

```bash
make data_process
```

### Text Processing

Text processing is crucial for extracting valuable information from text data. The `text_processing.py` script performs the following tasks:

- Loads the preprocessed data from `data/prep.csv`.
- Uses the KeyBERT library to extract keywords from the description column.
- Creates binary columns for unique keywords found in descriptions.
- Calculates the Spearman correlation between keywords and the target variable.
- Selects relevant columns based on correlation.
- Saves the processed data to `data/text_p.csv`.

To perform text processing, run:

```bash
make text_process
```

### Feature Engineering

Feature engineering is essential for creating meaningful features from the dataset. The `feature_engineering.py` script performs the following tasks:

- Loads the text-processed data from `data/text_p.csv`.
- Creates binary columns for amenities.
- Performs one-hot encoding for categorical columns.
- Imputes missing values using K-Nearest Neighbors (KNN) imputation.
- Drops rows with remaining missing values.
- Saves the processed data to `data/inpuded.csv`.

To perform feature engineering, run:

```bash
make new_features
```

### Outlier Detection

Outlier detection helps identify and handle extreme data points. The `outliers_detection.py` script performs the following tasks:

- Loads the feature-engineered data from `data/inpuded.csv`.
- Detects outliers using Isolation Forest, One-Class SVM, Z-Score, and Tukey's Method.
- Combines outlier results and removes outlier rows.
- Saves the data without outliers to `data/outlier_removed.csv`.

To detect and remove outliers, run:

```bash
make outliers
```

### Model Training

Model training involves selecting the best machine learning model for house price prediction. The `train.py` script performs the following tasks:

- Loads the data without outliers from `data/outlier_removed.csv`.
- Splits the dataset into training and testing sets.
- Trains several regression models, including CatBoost, XGBoost, LightGBM, and Random Forest.
- Selects the best-performing model based on Mean Squared Error (MSE).
- Saves the best model to `model/best_model.joblib`.

To train and evaluate models, run:

```bash
make train
```

## Results

The project evaluates multiple regression models and selects the best-performing model based on Mean Squared Error (MSE) and Mean Absolute Error (MAE). Here are the results:

- **Best Model Selected:** LGBMRegressor
- **Best Model MSE (Mean Squared Error):** 0.2905
- **Best Model MAE (Mean Absolute Error):** 0.4478
- **Best Model Saved:** [best_model.joblib](model/best_model.joblib)

### Model Comparison

Here's a comparison of the performance of different regression models:

| Model         | MSE      | MAE      |
|---------------|----------|----------|
| CatBoost      | 0.290520 | 0.447828 |
| XGBoost       | 0.193087 | 0.323286 |
| LightGBM      | 0.182430 | 0.316493 |
| Random Forest | 0.235428 | 0.364912 |
| Ensemble      | 0.188871 | 0.313804 |

The best model, LGBMRegressor, outperformed other models in terms of MSE and MAE. You can use the saved best model for making house price predictions.

