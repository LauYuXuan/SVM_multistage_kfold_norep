#!/bin/bash

# Check if the required argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <independent_data_file.csv>"
    exit 1
fi

independent_data_file=$1
output_file="output_$(date +%Y%m%d_%H%M%S).log"
python_script="output_script_$(date +%Y%m%d_%H%M%S).py"

# Write the Python code to a separate file
cat > "$python_script" <<EOF
# -*- coding: latin-1 -*-

import sys
import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from datetime import date


today = date.today()
print(today)  # Output: 2023-04-16

independent_data_file = sys.argv[1]


# Read the independent cohort file as a CSV file
independent_data = pd.read_excel(independent_data_file, index_col=0)
print("Shape of independent_data:", independent_data.shape)
print("Columns of independent_data:", independent_data.columns)
print(independent_data.head())

# Extract the unique labels from the column names
labels = ['HC', 'CRC0', 'CRC1']
print("Unique labels:", labels)

# Perform ROC analysis for each label
for label in labels:
    print(f"\nAnalysis for Label: {label}")
    
    # Create binary labels based on the current label
    binary_labels = independent_data.columns.str.contains(label).astype(int)
    
    # Apply SMOTE to the data
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(independent_data.T, binary_labels)
    
    # Perform feature ranking
    selector = SelectKBest(score_func=f_classif, k='all')
    selector.fit(X_resampled, y_resampled)
    selected_indices = selector.get_support(indices=True)

    if len(selected_indices) > 0:
        selected_features = independent_data.index[selected_indices]
        feature_scores = pd.DataFrame({'Feature': selected_features, 'Score': selector.scores_[selected_indices]})
        feature_scores = feature_scores.sort_values('Score', ascending=False)
        print("Feature Ranking:")
        print(feature_scores.head(10))  # Print top 10 features
    else:
        print("No features were selected for this label.")
    
    # Create an SVM classifier
    clf = SVC(kernel='linear', probability=True, random_state=42)
    
    # Split the data into training/testing (80%) and validation (20%) sets
    X_train_test, X_val, y_train_test, y_val = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    
    # Split the training/testing set into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X_train_test, y_train_test, test_size=0.2, random_state=42)
    
    # Train the classifier on the training data
    clf.fit(X_train, y_train)
    
    # Evaluate the classifier on the test data
    y_test_pred = clf.predict(X_test)
    print("Test Results:")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_test_pred))
    print("Accuracy:")
    print(accuracy_score(y_test, y_test_pred))
    
    # Calculate ROC curve and AUC for the test set
    y_test_pred_proba = clf.predict_proba(X_test)[:, 1]
    fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_test_pred_proba)
    roc_auc_test = auc(fpr_test, tpr_test)
    print("ROC AUC (Test Set):", roc_auc_test)
    
    # Evaluate the classifier on the validation data
    y_val_pred = clf.predict(X_val)
    
    # Calculate ROC curve and AUC for the validation set
    y_val_pred_proba = clf.predict_proba(X_val)[:, 1]
    fpr_val, tpr_val, thresholds_val = roc_curve(y_val, y_val_pred_proba)
    roc_auc_val = auc(fpr_val, tpr_val)
    print("ROC AUC (Validation Set):", roc_auc_val)
    
    # Plot ROC curves for both test and validation sets
    plt.figure()
    plt.plot(fpr_test, tpr_test, color='darkorange', lw=2, label='ROC curve (Test Set) (AUC = %0.2f)' % roc_auc_test)
    plt.plot(fpr_val, tpr_val, color='blue', lw=2, label='ROC curve (Validation Set) (AUC = %0.2f)' % roc_auc_val)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - Label {label}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(f'roc_curve_label_noRep{label}.png')
EOF

# Run the Python script with the provided argument and capture the output
python3 "$python_script" "$independent_data_file" | tee "$output_file"

# Remove the temporary Python script file
rm "$python_script"