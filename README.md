 Music Genre Classification using Machine Learning
 Overview

This project builds an end-to-end machine learning pipeline to classify music tracks into 11 genres using audio features such as danceability, energy, tempo, loudness, and acousticness.

The pipeline includes data preprocessing, feature engineering, model training, and evaluation across multiple machine learning algorithms. Performance comparison is conducted using Stratified K-Fold cross-validation to identify the best model.

 ML Pipeline Architecture
Raw Dataset
   │
   
Data Cleaning
(Remove duplicates, handle missing values)
   │
   
Exploratory Data Analysis (EDA)
(Distribution plots, correlations, pairplots)
   │
   
Feature Engineering
(Target Encoding, Frequency Encoding)
   │
   
Feature Scaling
(StandardScaler)
   │
   
Class Imbalance Handling
(SMOTE Oversampling)
   │
   
Model Training
(Random Forest, SVM, Neural Network, etc.)
   │
   
Model Evaluation
(Accuracy, Precision, Recall, F1)
   │
   
Best Model Selection
(Random Forest)
   │
   
Prediction Generation
(submission.csv)
 Dataset
Training Data

17,996 samples

17 features

Test Data

7,713 samples

16 features

Target Variable

Class → Music genre label (11 classes)

Important Features

Danceability

Energy

Loudness

Speechiness

Acousticness

Instrumentalness

Tempo

Valence

Duration

Time Signature

 Tech Stack
Programming

Python

Data Processing

Pandas

NumPy

Visualization

Matplotlib

Seaborn

Machine Learning

Scikit-learn

XGBoost

LightGBM

CatBoost

Deep Learning

PyTorch

Handling Imbalanced Data

SMOTE (Imbalanced-learn)

 Exploratory Data Analysis

EDA was performed to understand feature relationships and distributions.

Key visualizations include:

Genre distribution

Feature distribution plots

Correlation heatmaps

Pairplots for key audio features

Outlier detection using boxplots

Example visualization:

Genre Distribution
|████████████████████████|
|██████████████          |
|████████████████        |
 Feature Engineering

Several transformations were applied to improve model performance:

Encoding

Frequency Encoding for Track Name

Target Encoding for Artist Name and Track Name

Outlier Handling

Z-score method

Outliers replaced with median values

Feature Scaling

Standardization using StandardScaler

 Handling Class Imbalance

The dataset contained imbalanced genre classes.
To address this, the project uses:

SMOTE (Synthetic Minority Oversampling Technique)

This generates synthetic samples for minority classes during training.

 Models Implemented

The following models were trained and evaluated:

Neural Network (PyTorch)

Random Forest

Support Vector Machine (SVM)

Decision Tree

Logistic Regression

Naive Bayes

K-Nearest Neighbors (KNN)

Perceptron with Polynomial Features

Training was performed using Stratified K-Fold Cross Validation (3 folds).

 Model Performance
Model	Accuracy
Random Forest	0.8255
SVM (RBF Kernel)	0.8220
Neural Network	0.8095
Decision Tree	0.8046
Logistic Regression	0.7255
Naive Bayes	0.6956
Perceptron	0.6811
KNN	0.6138

 Best Model: Random Forest

 Feature Importance

Feature importance analysis was performed using the Random Forest model to determine the most influential audio features affecting genre classification.

Top contributing features include:

Energy

Loudness

Tempo

Danceability

Acousticness

 Project Structure
music-genre-classification
│
├── train.csv
├── test.csv
├── notebook.ipynb
│
├── submission.csv
│
├── plots
│   ├── correlation_heatmap.png
│   ├── feature_distribution.png
│   └── confusion_matrix.png
│
└── README.md
How to Run
 Install Dependencies
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn torch xgboost lightgbm catboost category_encoders
 Run the Notebook

Place the following files in the project directory:

train.csv
test.csv

Run the notebook or Python script to train models and generate predictions.

 Output

The project produces:

Model evaluation metrics

Confusion matrices

Feature importance plots

Final prediction file

submission.csv

The submission file contains one-hot encoded genre predictions for each test sample.

 Conclusion

This project demonstrates a complete machine learning workflow from data preprocessing to model deployment.

Among all evaluated models, Random Forest and SVM achieved the best performance, highlighting the effectiveness of ensemble methods for music genre classification.
