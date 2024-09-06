# evaluate_model.py
import json
import numpy as np
import argparse

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
    parser.add_argument('model_path', type=str, help='Model JSON file path.')
    parser.add_argument('data_path', type=str, help='Test data CSV file path.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    X_test, y_test = load_test_data(args.data_path)
    model_params = load_model(args.model_path)
    y_pred = predict(X_test, model_params)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
