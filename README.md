# Cancer-prediction-using-ML

**Breast Cancer Prediction using SVM**
This project implements a Support Vector Machine (SVM) model to predict the diagnosis of breast cancer based on features extracted from cell images. The model is trained on the PIMA Breast Cancer Dataset.

**Project Overview**
The goal of this project is to build a reliable machine learning model that can assist in the early detection of breast cancer. By analyzing various characteristics of cell nuclei, the model classifies a tumor as either benign (B) or malignant (M).

**Dataset**
The project utilizes the PIMA Breast Cancer Dataset, which contains features computed from digitized images of fine needle aspirate (FNA) of a breast mass. The features describe characteristics of the cell nuclei present in the image.

Source: https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset
Columns: The dataset includes various features such as radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension, calculated for the mean, standard error, and "worst" or largest mean values of each characteristic. The diagnosis column indicates whether the tumor is benign (B) or malignant (M).
Shape: The dataset contains 569 samples and 33 features (including the ID and diagnosis columns).

**Methodology**
1. Data Loading and Analysis: The dataset is loaded using pandas. Initial exploration includes checking the shape and descriptive statistics of the data.
2. Data Preprocessing:
  a) The 'Unnamed: 32' column, which contains missing values, is dropped.
  b) The features (X) and the target variable (Y) are separated.
  c) The target variable 'diagnosis' is the label to be predicted.
  d) Data standardization is performed using StandardScaler to ensure that all features have a        similar scale, which is important for SVM models.
3. Train-Test Split: The data is split into training and testing sets to evaluate the model's performance on unseen data. A stratify split is used to maintain the same proportion of benign and malignant cases in both sets.
4. Model Training: A Support Vector Machine (SVM) classifier with a linear kernel is initialized and trained on the standardized training data.
5. Model Evaluation: The model's performance is evaluated using the accuracy score on both the training and testing datasets.
   
**Results**
The trained SVM model achieved the following accuracy scores:

1. Training Data Accuracy: 98.9010989010989 %
2. Testing Data Accuracy: 96.49122807017544 %
These results indicate that the model performs well in classifying breast tumors based on the provided features.

**How to Run**
1. Clone this repository.
2. Make sure you have the necessary libraries installed (numpy, pandas, scikit-learn).
3. Ensure the Cancer_Data.csv file is in the correct directory (or update the file path in the code).
4. Run the Python script or Jupyter Notebook containing the code.
**Dependencies**
2. numpy
3. pandas
4. scikit-learn
