"""
Model Evaluation Script for Iris Classification
Loads test data and trained model to calculate accuracy and performance metrics
"""

import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

def evaluate_model():
    """
    Load test data and trained model to evaluate performance
    """
    # Load test data
    print("Loading test data from data/test.csv...")
    test_df = pd.read_csv('data/test.csv')
    
    # Split features and target
    X_test = test_df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    y_test = test_df['Species']
    
    print(f"Test set shape: {X_test.shape}")
    
    # Load the trained model and scaler
    print("\nLoading trained model and scaler...")
    model = joblib.load('models/model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    
    # Scale test features
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions
    print("Making predictions on test set...")
    y_pred = model.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Display classification report
    print("\nClassification Report:")
    print("-"*60)
    print(classification_report(y_test, y_pred))
    
    # Display confusion matrix
    print("Confusion Matrix:")
    print("-"*60)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    print("\n" + "="*60)
    print("Evaluation complete!")
    print("="*60)
    
    return accuracy

if __name__ == "__main__":
    evaluate_model()
