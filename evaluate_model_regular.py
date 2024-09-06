# evaluate_model.py
import json
import numpy as np
import argparse
import os
from datetime import datetime

def load_test_data(file_path):
    """Load test data from CSV."""
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    X_test = data[:, 0]
    y_test = data[:, 1]
    return X_test, y_test

def load_model(file_path):
    """Load model parameters from JSON."""
    with open(file_path, 'r') as file:
        return json.load(file)

def predict(X, model_params):
    """Predict using linear regression equation."""
    return model_params['slope'] * X + model_params['intercept']

def mean_squared_error(y_true, y_pred):
    """Calculate MSE."""
    return np.mean((y_true - y_pred) ** 2)

def parse_arguments():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description='Evaluate linear regression model.')
    parser.add_argument('model_folder_path', type=str, help='Model JSON file path.')
    parser.add_argument('data_path', type=str, help='Test data CSV file path.')
    return parser.parse_args()

def save_model(model_params, file_path):
    """Save model parameters to JSON."""
    with open(file_path, 'w') as file:
        json.dump(model_params, file)
    #print("Model saved to", file_path)

# 3. "Deploy" the best model at any time by saving model weights 
# to a production directory with version tracking.
if __name__ == "__main__":
    # python evalueate_model_regular.py model prod/test_data2.csv
    args = parse_arguments()
    X_test, y_test = load_test_data(args.data_path)
    models = os.listdir(args.model_folder_path)
    best_mse = float('inf')
    for model in models:
        model_params = load_model(f"{args.model_folder_path}/{model}")
        y_pred = predict(X_test, model_params)
        mse = mean_squared_error(y_test, y_pred)
        if mse < best_mse:
            best_mse = mse
            best_model = model
            best_model_params = model_params
        print(f"Model: {model}, Mean Squared Error: {mse}")
    print(f"Best Model: {best_model}, Best Mean Squared Error: {best_mse}")
    # Save best model
    save_model(best_model_params, f"prod/{best_model.split('.')[0]}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json")
    
