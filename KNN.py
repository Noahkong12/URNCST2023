import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
data = pd.read_csv('ML3.csv')
target = 'ARTH12'

# Remove the 'ID' column if it exists
if 'ID' in data.columns:
    data.drop('ID', axis=1, inplace=True)

# Separate the features (X) and the target variable (y)
X = data.drop(target, axis=1)
y = data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of SMOTE for oversampling
smote = SMOTE(random_state=42)

# Fit and apply SMOTE to the training data
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Create a KNN classifier model
knn_classifier = KNeighborsClassifier(n_neighbors=300)

# Train the model
knn_classifier.fit(X_train_scaled, y_train_resampled)

# Make predictions on the test data
y_pred = knn_classifier.predict(X_test_scaled)

# Calculate classification metrics
classification_report_str = classification_report(y_test, y_pred)
confusion_matrix_result = confusion_matrix(y_test, y_pred)
# Calculate classification metrics
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)
confusion_matrix_result = confusion_matrix(y_test, y_pred)

TP = confusion_matrix_result[1, 1]  # True Positives
FP = confusion_matrix_result[0, 1]  # False Positives
TN = confusion_matrix_result[0, 0]  # True Negatives
FN = confusion_matrix_result[1, 0]  # False Negatives

PPV = TP / (TP + FP)  # Positive Predictive Value
NPV = TN / (TN + FN)  # Negative Predictive Value
Sensitivity = TP / (TP + FN)  # Positive Predictive Value
Specificity = TN / (FN + TN)  # Positive Predictive Value

print(f"Positive Predictive Value (PPV): {PPV:.2f}")
print(f"Negative Predictive Value (NPV): {NPV:.2f}")
print(f"Sensitivity: {Sensitivity:.2f}")
print(f"Specificity: {Specificity:.2f}")

print(f"Accuracy: {accuracy}")
print("KNN Classification Report:\n", classification_report_str)
print("KNN Confusion Matrix:\n", confusion_matrix_result)
