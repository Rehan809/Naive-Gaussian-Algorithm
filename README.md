# Naive-Gaussian-Algorithm
# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Load your dataset
# Replace 'iris_dataset.csv' with the path to your uploaded dataset
data = pd.read_csv('iris_dataset.csv')
# Display the first few rows of the dataset
print(data.head())
# Separate the independent variables (features) and the dependent variable (target)
# Make sure to adjust column names based on your dataset
X = data[['sepal length', 'sepal width', 'petal length', 'petal width']]
y = data['species']  # 'species' is the target variable
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize the Gaussian Naive Bayes model
model = GaussianNB()
# Fit the model to the training data
model.fit(X_train, y_train)
# Predict species for the test set
y_pred = model.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
