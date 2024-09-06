import numpy as np
import csv
import json
import argparse

def load_data_from_csv(file_path):
    """Load data from CSV."""
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader, None)  # Skip header
        X, y = [], []
        for row in reader:
            X.append(float(row[0]))
            y.append(float(row[1]))
    return np.array(X), np.array(y)

def train_model(X, y):
    """Train linear regression model using np.linalg.lstsq."""
    X_b = np.c_[np.ones((len(X), 1)), X]  # Add bias term
    theta_best, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=None)
    return {'slope': theta_best[1], 'intercept': theta_best[0]}

def save_model(model_params, file_path):
    """Save model parameters to JSON."""
    with open(file_path, 'w') as file:
        json.dump(model_params, file)
    #print("Model saved to", file_path)

def parse_arguments():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description='Train and save a linear regression model.')
    parser.add_argument('data_path', type=str, help='Path to the input data CSV file.')
    parser.add_argument('model_path', type=str, help='Path to save the trained model JSON file.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    X, y = load_data_from_csv(args.data_path)
    model_params = train_model(X, y)
    save_model(model_params, args.model_path)
