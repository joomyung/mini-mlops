import numpy as np
import csv
import json
import argparse
import os

def check_most_recent_model(model_folder_path):
    """Return index for train."""
    models = os.listdir(model_folder_path)
    # sort the models by index
    models.sort(key=lambda x: int(x.split(".")[0].split("_")[1]))
    recent_model = models[-1]
    print(recent_model)
    recent_model_index = recent_model.split(".")[0].split("_")[1]
    index_to_train = int(recent_model_index) + 1
    return index_to_train

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
    parser.add_argument('data_folder_path', type=str, help='Path to the input data CSV file.')
    parser.add_argument('model_folder_path', type=str, help='Path to save the trained model JSON file.')
    return parser.parse_args()

if __name__ == "__main__":
    # python train_model_regular.py data model
    args = parse_arguments()
    index_to_train = check_most_recent_model(args.model_folder_path)
    print(index_to_train)
    while True:
        try:
            data_path = os.path.join(args.data_folder_path, f"data_{str(index_to_train)}.csv")
            print(data_path)
            X, y = load_data_from_csv(data_path)
            model_params = train_model(X, y)
            model_path = os.path.join(args.model_folder_path, f"model_{str(index_to_train)}.json")
            save_model(model_params, model_path)
            index_to_train += 1
        except Exception as e:
            print(e)
            break

