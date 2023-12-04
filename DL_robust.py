import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

random_seed = 42
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

def standardize_column_names(df, time_point):
    """
    Standardize column names by removing the specific time point prefix.
    """
    df.columns = [col.replace(f'V{time_point}', '') if col.startswith(f'V{time_point}') else col for col in df.columns]
    return df

# Load and preprocess the datasets
filenames = ['AllClinical03', 'AllClinical05', 'AllClinical06', 'AllClinical08', 'AllClinical10']
all_data = []

for filename in filenames:
    time_point = filename[-2:]
    df = pd.read_csv(f"{filename}_normalized.csv")
    df = standardize_column_names(df, time_point)
    all_data.append(df)

# Combine all datasets
combined_df = pd.concat(all_data).reset_index(drop=True)

# Shuffle the combined dataset
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Features and Labels
X = combined_df.drop(['ID', 'ARTH12'], axis=1)
y = combined_df['ARTH12']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Downsampling the Majority Class in Training Set
train_df = pd.concat([X_train, y_train], axis=1)
majority_class = train_df[train_df['ARTH12'] == 0]
minority_class = train_df[train_df['ARTH12'] == 1]

majority_downsampled = resample(majority_class,
                                replace=False,  # sample without replacement
                                n_samples=len(minority_class),  # to match minority class size
                                random_state=123)  # for reproducibility

train_df_balanced = pd.concat([majority_downsampled, minority_class])

# Splitting Features and Labels for Balanced Dataset
X_train_balanced = train_df_balanced.drop('ARTH12', axis=1)
y_train_balanced = train_df_balanced['ARTH12']

# Scale features
scaler = StandardScaler()
X_train_balanced_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

model = Sequential()
model.add(Dense(64, input_dim=X_train_balanced.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model with binary cross-entropy loss
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train_balanced, y_train_balanced, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy}")


# Evaluate the model on the test data
y_pred_prob = model.predict(X_test_scaled)
y_pred_binary = (y_pred_prob >= 0.5).astype(int)

# Calculate classification metrics
classification_report_str = classification_report(y_test, y_pred_binary)
confusion_matrix_result = confusion_matrix(y_test, y_pred_binary)

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

plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
labels = ['Non-diseased', 'Diseased']
plt.xticks([0.5, 1.5], labels)
plt.yticks([0.5, 1.5], labels, rotation=0)  # rotation=0 for horizontal y labels
plt.show()


# Calculate ROC curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

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

