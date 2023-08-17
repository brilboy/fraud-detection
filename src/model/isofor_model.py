# Load Data
import pandas as pd
data = pd.read_csv('src\data\creditcard.csv')

from sklearn.ensemble import IsolationForest
import joblib

# Splitting features and labels
X = data.drop('Class', axis=1)
y = data['Class']

annomalies = data[data['Class']==1]
normal = data[data['Class']==0]

outlier_fraction = len(annomalies)/float(len(normal))
outlier_fraction

# Initialize and train the Isolation Forest model
iso_forest = IsolationForest(contamination=outlier_fraction, n_estimators=100, max_samples=len(X), random_state=42, verbose=0)  # Adjust contamination based on the fraud rate
iso_forest.fit(X)

# Predict anomalies
scores_prediction = iso_forest.decision_function(X)
predictions_isofor = iso_forest.predict(X)

# Convert predictions to binary values: 1 for normal, -1 for anomaly
binary_predictions_isofor = [0 if prediction == 1 else 0 for prediction in predictions_isofor]

## Summary of Model Result
from sklearn.metrics import accuracy_score, classification_report

# Calculate metrics for model
accuracy = accuracy_score(y, binary_predictions_isofor)
classification_rep = classification_report(y, binary_predictions_isofor)

# Print model performance summary
print("Isolation Forest:")
print(f"Accuracy: {accuracy}")
print(f"Classification Report: {classification_rep}\n")

# Save the best Isolation Forest model
joblib.dump(iso_forest, r'src\result\isofor_model.joblib')