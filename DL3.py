
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

data = pd.read_csv('AllClinical03_normalized.csv')
target = 'V03ARTH12'
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

# Build a simple binary classification model
model = Sequential()
model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model with binary cross-entropy loss
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model on the oversampled training data
model.fit(X_train_scaled, y_train_resampled, epochs=50, batch_size=32, verbose=1)

# Evaluate the model on the test data
y_pred_prob = model.predict(X_test_scaled)
y_pred_binary = (y_pred_prob >= 0.5).astype(int)

# Calculate classification metrics
classification_report_str = classification_report(y_test, y_pred_binary)
confusion_matrix_result = confusion_matrix(y_test, y_pred_binary)

print("Classification Report:\n", classification_report_str)
print("Confusion Matrix:\n", confusion_matrix_result)

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

print("Classification Report:\n", classification_report_str)
print("Confusion Matrix:\n", confusion_matrix_result)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_result, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix of Model Trained by 2009 Data')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
labels = ['Non-diseased', 'Diseased']
plt.xticks([0.5, 1.5], labels)
plt.yticks([0.5, 1.5], labels, rotation=0)  # rotation=0 for horizontal y labels
plt.show()

# Calculate ROC curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Calculate precision-recall curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
