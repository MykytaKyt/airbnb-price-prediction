# House Price Prediction Project

This project is designed for predicting house prices using machine learning models. It includes data processing, text processing, feature engineering, outlier detection, model training, and evaluation.

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

Before you begin, ensure you have the following prerequisites installed:

- Python 3.x
- pip (Python package manager)

### Getting the Data

To get the newer data for this project, follow these steps:

1. Visit the [Inside Airbnb](http://insideairbnb.com/get-the-data/) website.
2. Download the dataset archive relevant to your project.
3. Extract the downloaded archive to the project directory.

Now you should have a CSV file named `train.csv` in the `data` directory, containing the data for house price prediction.

Or you can use data from `train.zip`, you need to unzip it first.

## Installation

To install the required Python libraries and dependencies, use the following command:

```bash
pip install -r requirements.txt
```

## Usage

This project is organized using a `Makefile` to simplify common tasks. Here's how to use it:

To run whole pipeline just use:

```bash
make all
```
### Data Processing

To preprocess the data, run the following command:

```bash
make data_process
```

### Text Processing

To perform text processing and keyword extraction, run the following command:

```bash
make text_process
```

### Feature Engineering

To perform feature engineering, including creating binary columns for amenities, run the following command:

```bash
make new_features
```

### Outlier Detection

To detect and remove outliers from the dataset, run the following command:

```bash
make outliers
```

### Model Training

To train and evaluate regression models, select the best-performing model, and save it, run the following command:

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
