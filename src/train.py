"""
Model Training Script for Iris Classification
Loads training data, trains a Logistic Regression model, and saves it
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os

def train_model():
    """
    Load training data, train a Logistic Regression model, and save it
    """
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Load training data
    print("Loading training data from data/train.csv...")
    train_df = pd.read_csv('data/train.csv')
    
    # Split features and target
    X_train = train_df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    y_train = train_df['Species']
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Number of classes: {y_train.nunique()}")
    
    # Feature scaling
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train Logistic Regression model
    print("Training Logistic Regression model...")
    model = LogisticRegression(
        max_iter=200,
        random_state=42,
        multi_class='ovr',
        solver='lbfgs'
    )
    model.fit(X_train_scaled, y_train)
    
    # Calculate training accuracy
    train_accuracy = model.score(X_train_scaled, y_train)
    print(f"\nTraining accuracy: {train_accuracy:.4f}")
    
    # Save the model and scaler
    model_path = 'models/model.joblib'
    scaler_path = 'models/scaler.joblib'
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"\nModel saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    print("\nModel training complete!")

if __name__ == "__main__":
    train_model()
