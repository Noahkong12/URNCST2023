import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

data = pd.read_csv('ML3.csv')
target = 'ARTH12'
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

# Create a Random Forest Classifier model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_classifier.fit(X_train_scaled, y_train_resampled)

# Make predictions on the test data
y_pred = rf_classifier.predict(X_test_scaled)

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
print("Classification Report:\n", classification_report_str)
print("Confusion Matrix:\n", confusion_matrix_result)


import shap
explainer = shap.TreeExplainer(rf_classifier)

# Generate SHAP values for the test set
shap_values = explainer.shap_values(X_test_scaled)

# Since shap_values is a list of arrays (one array for each class), we need to determine which one to use
# We will use the values for the positive class if it's a binary classification
# If it's a multi-class classification, choose the index appropriately
if len(shap_values) == 2:  # Binary classification
    shap_values = shap_values[1]  # This is for the positive class
else:
        # Multi-class classification: choose the index for the class of interest
        # shap_values = shap_values[class_index]
    pass

    # Calculate the mean absolute SHAP values for each feature
shap_sum = np.abs(shap_values).mean(axis=0)

    # Get the feature names from the dataset
feature_names = X.columns.tolist()

    # Combine feature names and their corresponding mean absolute SHAP values
feature_importance = pd.DataFrame(list(zip(feature_names, shap_sum)),
                                    columns=['feature_name', 'shap_value'])

    # Sort the features based on mean absolute SHAP values
feature_importance.sort_values(by=['shap_value'], ascending=False, inplace=True)

    # Get the top 3 features
    # Plot the SHAP summary plots separately
TF = feature_importance.head(3)['feature_name'].tolist() #TF denotes Top Features
OF = feature_importance.head(23)['feature_name'].tolist()
OF = OF[3:]
top_features1 = ["Time to Complete 20m Walk", " Blood pressure: Systolic (mm Hg)", "Currently work for pay(Yes/No)"]
OtherFeatures = ["Currently Work for Pay(Yes/No)", "Repeated Chair Stands Time(Trial 2)", "20m Walk: Number of Steps", 
     "Outdoor Gardening in Past 7 Days", "Repeated Chair Stands Time(Trial1)", "Leisure Activities: Muscle Strength/Endurance", 
     "Gout Diagnosis by Physician: Past 12 Months", "Time to Complete 20m Walk(Trial 2)",
     "Number of hours worked(Past 7 Days)", "Lawn Work in Past 7 Days", "Age", "Weight", "Home Repairs", 
     "Moderate Sport/Cecreation(Hrs/Day)",
     "CES-D Score", "Heavy Houseworks", "20m Walk: Left Knee Pain(0-10)", "Strenuou Sport/Recreation(Hrs/Day)", 
     "Blood Pressure: Cuff Size Used", "Work (Pay/Volunteer)"
     ]
    
OtherFeatures = ["CES-D Score", "Walking, Past 7 Days", "Num of Falling in Past 12 Months", "20m Walk: Number of Steps(Trial2)",
                     "20m Walk: Max Left Knee Pain(0-10)", "Walking(Hrs/Day)", "20m Walk: Number of Steps(Trial1)", 
                     "20m Walk: Left Knee Pain(0-10)", "20m Walk: Right Knee Pain(0-10)", "Repeated Chair Stands Time(Trial 2)",
                     "Lawn Work Past 7 Days", "Repeated Chair Stands Time(Trial 1)", "20m Walk: Max Right Knee Pain(0-10)",
                     "Age", "Hours Worked, Past 7 Days", "Occupational Activity Level", "Blood Pressure: What Cuff Size Used",
                     "Blood pressure: diastolic (mm Hg)", "Outdoor Gardening, Past 7 Days", "Strenuous Sport/Recreation(hrs/day)"
     ]
    
     # Separate the SHAP values for the top 3 features
top_shap_values = [shap_values[:, feature_names.index(feature)] for feature in TF]
other_shap_values = [shap_values[:, feature_names.index(feature)] for feature in OF]


    # Summary plot for the top 3 features
    # shap.summary_plot(np.array(top_shap_values).T, X_test_scaled[:, [feature_names.index(f) for f in top_features]],
    #                 feature_names=top_features)
shap.summary_plot(np.array(top_shap_values).T, X_test_scaled[:, [feature_names.index(f) for f in TF]],
                     feature_names=top_features1)


    # Summary plot for the rest of the features
    # shap.summary_plot(np.array(other_shap_values).T, X_test_scaled[:, [feature_names.index(f) for f in feature_names if f not in top_features]],
    #                 feature_names=[f for f in feature_names if f not in top_features])
shap.summary_plot(np.array(other_shap_values).T, X_test_scaled[:, [feature_names.index(f) for f in OF]],
                    feature_names = OtherFeatures)
    
 