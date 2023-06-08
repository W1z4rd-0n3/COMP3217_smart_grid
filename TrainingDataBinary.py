# Random Forest algorithm for TrainingDataBinary

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns


# Load the data into a pandas DataFrame
data = pd.read_csv('TrainingDataBinary.csv')

# Splitting data into features and labels
X = data.iloc[:, :-1]  # Features are all columns except the last one
y = data.iloc[:, -1]   # Last column is Labels

# Splitting data into training and validation datasets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=10)

# Normalize the data by applying standardization and normalization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
transformer = Normalizer().fit(X_train)
transformer.transform(X_train)

# Train the model
model = RandomForestClassifier(n_estimators= 200, criterion = "entropy")
model_pred = RandomForestClassifier(n_estimators= 100, criterion = 'entropy')
model.fit(X_train, y_train)
model_pred.fit(X, y)

# Make predictions on validation data
y_pred = model.predict(X_val)
y_pred2 = model_pred.predict(X)

# Calculate accuracy
accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy: {accuracy}")
accuracy_pred = accuracy_score(y, y_pred2)
print(f"Accuracy of predictions: {accuracy_pred}")

# Calculate F1 score
f1 = f1_score(y_val, y_pred, average='macro')
print(f"F1 score: {f1}")

# Create confusion matrix
cm = confusion_matrix(y_val, y_pred)
print("Confusion Matrix:")
print(cm)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Create classification report
report = classification_report(y_val, y_pred)
print("Classification Report:")
print(report)

# Plot the data points
plt.figure(figsize=(8, 6))
plt.scatter(range(len(y_val)), y_val, c='b', label='Actual')
plt.scatter(range(len(y_pred)), y_pred, c='r', label='Predicted')
plt.xlabel('Sample')
plt.ylabel('Label')
plt.title('Actual vs Predicted Labels')
plt.legend()
plt.show()

# Read the testing data
test_data = pd.read_csv("TestingDataBinary.csv", header=None)

# Assigning features for testing
x_test = test_data.iloc[:, :].values

# Normalize the data by applying standardization 
x_test = scaler.transform(test_data)
print(x_test)

# Applying trained model to test data for identification
test_predictions = model.predict(x_test)
print(test_predictions)

# Output predicted labels to test dataset and writing them into a csv file
test_data['test_predictions'] = test_predictions
test_data.to_csv('TestingResultsBinary.csv', index=False, header=False)

