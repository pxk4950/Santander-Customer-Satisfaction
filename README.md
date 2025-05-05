# Santander Customer Satisfaction


This repository contains an attempt to predict customer satisfaction for Santander Bank using data from the [Santander Customer Satisfaction Kaggle Challenge](https://www.kaggle.com/competitions/santander-customer-satisfaction/overview).



## Overview

<<<<<<< HEAD
The objective of the Kaggle challenge is to predict customer dissatisfaction using anonymized numerical features representing customer behavior. The dataset consists of more than 76,000 customer records and a binary target variable indicating satisfaction. The task was approached as a **binary classification problem**. Data preprocessing, exploration, and cleaning were conducted, followed by training and evaluation of machine learning model, **Random Forests, and XGBoost**. The imbalance in the dataset was addressed through class weighting. Feature selection techniques were applied to enhance generalization. The best performance was achieved using an XGBoost model with an accuracy of 96 percentage and a AUC-score of 0.81. At this time, the heighest AUC-score on Kaggle Subnission is 0.82.
=======
The objective of the Kaggle challenge is to predict customer dissatisfaction using anonymized numerical features representing customer behavior. The dataset consists of more than 76,000 customer records and a binary target variable indicating satisfaction. The task was approached as a **binary classification problem**. Data preprocessing, exploration, and cleaning were conducted, followed by training and evaluation of machine learning model, **Random Forests, and XGBoost**. The imbalance in the dataset was addressed through class weighting. Feature selection techniques were applied to enhance generalization. The best performance was achieved using an XGBoost model with an accuracy of 96 percentage and a log loss of 0.14.
>>>>>>> a234238f312e4332f1c9febd1bab46d284f3c496

## Summary of Workdone

### Data

- **Input**: CSV file with 370 anonymized numerical features and a binary target variable.
- **Dataset Size**: Approximately 76,000 training instances.
- **Train/Test Split**: A stratified 80/20 split was applied.
  
### Preprocessing:
  - Constant and duplicate columns were removed.
  - Class imbalance was identified and addressed using class weights.

### Data Visualization

- Visualizations included:
  - Histogram of all features
  - Class distribution (notably imbalanced)
  - ROC curve of Random Forest and Xgboost Models



## Problem Formulation

- **Input**: anonymized numerical features.
- **Output**: Binary prediction (0 = satisfied, 1 = dissatisfied).

### Models

The following models were developed and evaluated:

- Random Forest
- XGBoost Classifier

Theese models are generally used for the classification problems and as this was a classification problem too, I decided to train on theese models.

Evaluation metrics included `accuracy`, `log loss`, `precision`, `recall`, and `ROC-AUC`.



##  Training

- **Environment**: Python 3.11, Jupyter Notebook.
- **Libraries**: Scikit-learn, XGBoost, Pandas, Numpy, Tabulate
- **Challenges**:
  - Class imbalance  
    There was very high class imbalance. This was handled by comparing evaluation metrics (accuracy, log loss, precision,  recall, and ROC-AUC) of the different models like SMOTE, Oversampling, Baseline, Class-weights and undersampling. These gave the conlusion that best result is when the class imbalance is handled with class weighting.
  - Redundant features were filtered out looking at the duplicated in feaatures and constant data in the features.


## Performance Comparison

| Model              | Accuracy | Log Loss | ROC-AUC |
|-------------------|----------|----------|---------|
| Random Forest       | 0.9551     | 0.3330     | 0.7629    |
| XGBoost             | 0.9607     | 0.1410     | 0.8110    |

Performance metrics show that XGBoost consistently outperformed the random Forest Model in accuracy and log loss.



##  Conclusions

XGBoost was found to be the most effective model for this classification task. Its ability to capture non-linear relationships and handle class imbalance contributed to superior performance. 
<<<<<<< HEAD

## How to reproduce results
To reproduce the results:

1. Download the dataset from the [Santander Customer Satisfaction Kaggle Challenge](https://www.kaggle.com/competitions/santander-customer-satisfaction/data).
2. Install all required libraries and modules as described in the [`Software Setup`](#software-setup) section.
3. Run the notebook [`DataCleaning.ipynb`](DataCleaning.ipynb) to preprocess and clean the data.
4. Run the notebook [`MachineLearning.ipynb`](MachineLearning.ipynb) to train models and evaluate results.
5. # Execute [`DataVisualization.ipynb`](DataVisualization.ipynb) to see visualizations
=======
---
## How to reproduce results
To reproduce the results:

1. Download the dataset from the [Santander Customer Satisfaction Kaggle Challenge](https://www.kaggle.com/competitions/santander-customer-satisfaction/data).
2. Install all required libraries and modules as described in the Software Setup section.
3. Run the notebook [`Initial Look and Data Cleaning.ipynb`](Initial%20Look%20and%20Data%20Cleaning.ipynb) to preprocess and clean the data.
4. Execute [`Data Visualization and Machine Learning.ipynb`](Data%20Visualization%20and%20Machine%20Learning.ipynb) to train models and evaluate results.
>>>>>>> a234238f312e4332f1c9febd1bab46d284f3c496
   
## Future Work

- Application of advanced resampling techniques.
- Application of Feature importance for better results.
- Exploration of deep learning models.

<<<<<<< HEAD
## Overview of Files and Folders in Repository

- [README.md](README.md): A general overview of the project, including background, methodology, and results.
- [DataCleaning.ipynb](DataCleaning.ipynb): Data loading, preprocessing, and cleaning are performed in this notebook. Constant and duplicate features are removed, and class imbalance is addressed.
- [MachineLearning.ipynb](MachineLearning.ipynb): Random Forest and XGBoost models are trained, evaluated, and applied to the test dataset.
- [DataVisualization.ipynb](DataVisualization.ipynb): Exploratory data analysis is conducted through feature histograms. 
=======
---
## Overview of Files in Repository

- [README.md](README.md): A general overview of the project, including background, methodology, and results.
- [Initial Look and Data Cleaning.ipynb](Initial%20Look%20and%20Data%20Cleaning.ipynb): Data loading, preprocessing, and cleaning are performed in this notebook. Constant and duplicate features are removed, and class imbalance is addressed.
- [Data Visualization and Machine Learning.ipynb](Data%20Visualization%20and%20Machine%20Learning.ipynb): Exploratory data analysis is conducted through feature histograms. Random Forest and XGBoost models are trained, evaluated, and applied to the test dataset.
>>>>>>> a234238f312e4332f1c9febd1bab46d284f3c496

## Software Setup

The project was developed using Python 3.10+. The following packages are required for successful execution of the notebooks:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`
- `jupyter`

