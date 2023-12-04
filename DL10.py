
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

data = pd.read_csv('AllClinical10_normalized.csv')
target = 'V10ARTH12'

if 'ID' in data.columns:
    data.drop('ID', axis=1, inplace=True)

X = data.drop(target, axis=1)
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
smote = SMOTE(random_state=42)

X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

model = Sequential()
model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

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

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_result, annot=True, fmt='d', cmap='Blues')

# Set the title and labels for the heatmap
plt.title('Confusion Matrix of Model Trained by 2016(Nov 30th) Data')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')

# Define the labels
labels = ['Non-diseased', 'Diseased']

# Set the tick labels
plt.xticks([0.5, 1.5], labels)
plt.yticks([0.5, 1.5], labels, rotation=0)  # rotation=0 for horizontal y labels

# Show the plot
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

import shap
import matplotlib.pyplot as plt
together = False 
if not together:
    # Create a SHAP explainer object for your model
    explainer = shap.DeepExplainer(model, X_train_scaled)

    # Generate SHAP values for the test set
    shap_values = explainer.shap_values(X_test_scaled)

    # Get the mean absolute SHAP values for each feature
    shap_sum = np.abs(shap_values[0]).mean(axis=0)

    # Get the feature names from the dataset
    feature_names = X.columns.tolist()

    # Combine feature names and their corresponding mean absolute SHAP values
    feature_importance = pd.DataFrame(list(zip(feature_names, shap_sum)),
                                    columns=['feature_name', 'shap_value'])

    # Sort the features based on mean absolute SHAP values
    feature_importance.sort_values(by=['shap_value'], ascending=False, inplace=True)

    # Get the top 3 features
    TF = feature_importance.head(3)['feature_name'].tolist()
    OF = feature_importance.head(23)['feature_name'].tolist()
    OF = OF[3:]
    top_features1 = ["Time to Complete 20m Walk", "Systolic Blood Pressure (mmHg)", "Num of Falling in Past 12 Months"]
    OtherFeatures = ["Currently Work for Pay(Yes/No)", "Repeated Chair Stands Time(Trial 2)", "20m Walk: Number of Steps", 
     "Outdoor Gardening in Past 7 Days", "Repeated Chair Stands Time(Trial1)", "Leisure Activities: Muscle Strength/Endurance", 
     "Gout Diagnosis by Physician: Past 12 Months", "Time to Complete 20m Walk(Trial 2)",
     "Number of hours worked(Past 7 Days)", "Lawn Work in Past 7 Days", "Age", "Weight", "Home Repairs", 
     "Moderate Sport/Cecreation(Hrs/Day)",
     "CES-D Score", "Heavy Houseworks", "20m Walk: Left Knee Pain", "Strenuou Sport/Recreation(Hrs/Day)", 
     "Blood Pressure: Cuff Size Used", "Work (Pay/Volunteer)"
     ]
    

    # Separate the SHAP values and feature names for the top 3 features
    top_shap_values = [shap_values[0][:, feature_names.index(feature)] for feature in TF]
    # other_shap_values = [shap_values[0][:, feature_names.index(feature)] for feature in feature_names if feature not in TF]
    other_shap_values = [shap_values[0][:, feature_names.index(feature)] for feature in OF]

    # Now we can plot the SHAP summary plots separately

    # Summary plot for the top 3 features
    shap.summary_plot(np.array(top_shap_values).T, X_test_scaled[:, [feature_names.index(f) for f in TF]],
                    feature_names=top_features1)

    # Summary plot for the rest of the features
    shap.summary_plot(np.array(other_shap_values).T, X_test_scaled[:, [feature_names.index(f) for f in OF]],
                    feature_names=OtherFeatures)

else:
        # Create a SHAP explainer object. 
    explainer = shap.DeepExplainer(model, X_train_scaled)

    # Generate SHAP values for the test set
    shap_values = explainer.shap_values(X_test_scaled)

    # Visualize the first prediction's explanation
    shap.initjs()  # Initialize JavaScript in the notebook for SHAP plots
    shap.force_plot(explainer.expected_value[0], shap_values[0][0], X_test_scaled[0])

    # You can also create summary plots
    shap.summary_plot(shap_values, X_test_scaled)

    explainer = shap.DeepExplainer(model, X_train_scaled)

    feature_names = X.columns.tolist()

    # Create a SHAP explainer object
    explainer = shap.DeepExplainer(model, X_train_scaled)

    # Generate SHAP values for the test set
    shap_values = explainer.shap_values(X_test_scaled)

    # Visualize the first prediction's explanation with feature names
    shap.force_plot(explainer.expected_value[0], shap_values[0][0], feature_names=feature_names)

    # Create a summary plot with feature names
    shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_names)