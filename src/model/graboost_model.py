import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load Data
data = pd.read_csv('src\data\creditcard.csv')

# Splitting features and labels
X = data.drop('Class', axis=1)
y = data['Class']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

# Initialize the GradientBoostingClassifier
graboost_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, subsample=0.8, random_state=42)

# Fit the model
graboost_model.fit(X_train, y_train)

# Predict on the test set
y_pred = graboost_model.predict(X_test)

# Calculate metrics for model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Print model performance summary
print("Isolation Forest:")
print(f"Accuracy: {accuracy}")
print(f"Classification Report: {classification_rep}\n")

# Save the best Isolation Forest model
joblib.dump(graboost_model, r'src\result\graboost_model.joblib')