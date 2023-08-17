import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils import compute_class_weight
from tensorflow.keras.callbacks import ModelCheckpoint

# Load Data
data = pd.read_csv('src\data\creditcard.csv')

# Splitting features and labels
X = data.drop('Class', axis=1)
y = data['Class']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Calculate class weights for handling class imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = {0: class_weights[0], 1: class_weights[1]}

# Define a function to build and compile the model
def build_model(input_dim):
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_dim=input_dim),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Initialize ModelCheckpoint callback to save the best model
checkpoint = ModelCheckpoint(
    r'src\result\neunet_model.h5', 
    monitor='val_accuracy',  # You can also use 'val_loss' here
    mode='min', 
    save_best_only=True,
    verbose=1
)

# Loop through different hyperparameter configurations and evaluate
for hidden_units in [32, 64, 128]:
    for dropout_rate in [0.2, 0.3, 0.4]:
        for batch_size in [32, 64]:
            model = build_model(input_dim=X_train.shape[1])
            print(f"Training with hidden units: {hidden_units}, dropout rate: {dropout_rate}, batch size: {batch_size}")
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=10,
                batch_size=batch_size,
                class_weight=class_weights,
                verbose=0,
                callbacks=[checkpoint]  # Add the ModelCheckpoint callback
            )
            
            y_pred = model.predict(X_test)
            y_pred_classes = np.round(y_pred)
            
            accuracy = accuracy_score(y_test, y_pred_classes)
            print(f"Accuracy: {accuracy:.4f}")
            
            print("Classification Report:")
            print(classification_report(y_test, y_pred_classes))
            print("=" * 80)