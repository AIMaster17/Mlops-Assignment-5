"""
Flask Web Application for Iris Species Prediction
A simple web interface for predicting iris flower species based on measurements
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained model and scaler
MODEL_PATH = 'models/model.joblib'
SCALER_PATH = 'models/scaler.joblib'

# Global variables for model and scaler
model = None
scaler = None

def load_model_and_scaler():
    """Load the model and scaler at startup"""
    global model, scaler
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            print("Model and scaler loaded successfully!")
        else:
            print("Warning: Model or scaler files not found. Please train the model first.")
    except Exception as e:
        print(f"Error loading model: {e}")

# Load model at startup
load_model_and_scaler()

@app.route('/')
def home():
    """Render the home page with the prediction form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests
    Expects four flower measurements and returns the predicted species
    """
    try:
        # Check if model is loaded
        if model is None or scaler is None:
            return jsonify({
                'error': 'Model not loaded. Please train the model first.'
            }), 500
        
        # Get input data from the form
        sepal_length = float(request.form.get('sepal_length'))
        sepal_width = float(request.form.get('sepal_width'))
        petal_length = float(request.form.get('petal_length'))
        petal_width = float(request.form.get('petal_width'))
        
        # Validate inputs
        if any(val < 0 for val in [sepal_length, sepal_width, petal_length, petal_width]):
            return render_template('index.html', 
                                 error='All measurements must be positive numbers.')
        
        # Create feature array
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Scale the features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        # Get prediction probability
        probabilities = model.predict_proba(features_scaled)[0]
        confidence = max(probabilities) * 100
        
        # Format the prediction for display
        species_name = prediction.replace('Iris-', '').capitalize()
        
        return render_template('index.html',
                             prediction=species_name,
                             confidence=f"{confidence:.2f}",
                             sepal_length=sepal_length,
                             sepal_width=sepal_width,
                             petal_length=petal_length,
                             petal_width=petal_width)
    
    except ValueError:
        return render_template('index.html',
                             error='Invalid input. Please enter valid numbers.')
    except Exception as e:
        return render_template('index.html',
                             error=f'An error occurred: {str(e)}')

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None and scaler is not None
    })

if __name__ == '__main__':
    # Run the Flask app
    # host='0.0.0.0' allows access from outside the container
    app.run(host='0.0.0.0', port=5000, debug=True)
