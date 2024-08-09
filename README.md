# heartDiseases
Heart Disease Analysis and Prediction
Authored by saeed asle

# Description
This repository contains a comprehensive analysis and predictive modeling of heart disease data. The project utilizes various machine learning techniques to predict the presence of heart disease in patients based on several features.

# Dataset
The dataset used in this analysis is the Heart 2020 Cleaned dataset. It contains various health-related attributes and a target column indicating the presence or absence of heart disease.

# Installation
To run this code, you need to have Python installed on your system along with the following libraries:
* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
You can install these libraries using pip:

      pip install pandas numpy matplotlib seaborn scikit-learn
  
# Project Structure
* Data Loading and Cleaning: The data is loaded from a CSV file and cleaned by removing duplicates and checking for null values.
* Exploratory Data Analysis (EDA): Various plots are generated to understand the distribution of features and their relationship with heart disease.
* Feature Engineering: Numerical and categorical features are processed, and dummy variables are created for categorical data.
* Modeling:
  * Logistic Regression: Both polynomial features and PCA are applied before fitting the logistic regression model.
  * Decision Tree: A grid search is performed to find the best hyperparameters.
  * Random Forest: The model is trained, and feature importance is analyzed.
  * Support Vector Machine (SVM): A grid search is performed to optimize the SVM model.
  * K-Nearest Neighbors (KNN): The model is fine-tuned using grid search.

# Usage
To run the analysis and prediction models, simply execute the code in a Python environment. The dataset file should be placed in the specified directory.

    python heart_disease.py
    
# Results
The project evaluates various models using accuracy scores on training and test data. The classification reports are also provided to assess the performance of each model in detail.

# Visualization
Several plots and visualizations are generated throughout the analysis to provide insights into the data, including:

Histograms and KDE plots
* Pie charts for categorical distribution
* Count plots for categorical features
* Feature importance bar charts

# Contributing
Feel free to fork this repository and submit pull requests if you'd like to contribute or improve the code.
