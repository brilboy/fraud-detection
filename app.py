from flask import Flask, request, render_template, jsonify
import numpy as np
import joblib
import tensorflow as tf
from keras.models import load_model

app = Flask(__name__)

# Load the saved models
best_iso_forest_model = joblib.load(r'src\result\isofor_model.joblib')
best_gb_model = joblib.load(r'src\result\graboost_model.joblib')
best_nn_model = load_model('src/result/neunet_model.h5')

@app.route('/')
def index():
    return render_template('predict.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        # Get user input
        time = float(request.form['Time'])
        amount = float(request.form['Amount'])

        # Generate other features randomly between -120 and 130
        other_features = np.random.uniform(-120, 130, 28)

        # Perform predictions with the models
        iso_prediction = best_iso_forest_model.predict([[time, amount, *other_features]])[0]
        gb_prediction = best_gb_model.predict([[time, amount, *other_features]])[0]
        
        # Use predict method for Keras model and then get the index with highest probability
        nn_prediction_probabilities = best_nn_model.predict(np.array([[time, amount, *other_features]]))
        nn_prediction = np.argmax(nn_prediction_probabilities, axis=-1)[0]

        # Convert prediction results to human-readable labels
        iso_label = "Fraudulent Transaction" if iso_prediction == 1 else "Normal Transaction"
        gb_label = "Fraudulent Transaction" if gb_prediction == 1 else "Normal Transaction"
        nn_label = "Fraudulent Transaction" if nn_prediction == 1 else "Normal Transaction"

        return render_template('predict.html', iso_label=iso_label, gb_label=gb_label, nn_label=nn_label)
    
    return render_template('predict.html')

if __name__ == '__main__':
    app.run()
