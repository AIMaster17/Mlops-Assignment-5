"""
Data Preparation Script for Iris Classification
Loads the Iris dataset and splits it into training and testing sets
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import os

def prepare_data():
    """
    Load the Iris dataset from the Dataset folder and split into train/test sets
    """
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Load the Iris dataset from the Dataset folder
    print("Loading Iris dataset from Dataset/Iris.csv...")
    df = pd.read_csv('Dataset/Iris.csv')
    
    # Display dataset info
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Split into features and target
    X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    y = df['Species']
    
    # Split the data into training and testing sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create train and test dataframes
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    # Save the splits
    train_path = 'data/train.csv'
    test_path = 'data/test.csv'
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\nData preparation complete!")
    print(f"Training set: {train_df.shape[0]} samples saved to {train_path}")
    print(f"Testing set: {test_df.shape[0]} samples saved to {test_path}")
    print(f"\nClass distribution in training set:")
    print(y_train.value_counts())

if __name__ == "__main__":
    prepare_data()
