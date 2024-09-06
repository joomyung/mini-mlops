import numpy as np
import csv
import argparse
import os

def generate_data():
    """Generate linear data with noise."""
    X = np.linspace(0, 10, 100)
    true_slope = 2.5
    true_intercept = -1.0
    noise = np.random.randn(100) * 2
    y = true_slope * X + true_intercept + noise
    return X, y

def save_data_to_csv(X, y, data_path):
    """Save data to CSV.""" 
    with open(data_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['X', 'y'])
        for x_val, y_val in zip(X, y):
            writer.writerow([x_val, y_val])
    #print("Data saved to", data_path)

def parse_arguments():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description='Generate linear data with noise.')
    parser.add_argument('data_path', type=str, help='Output data CSV file path.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    X, y = generate_data()
    save_data_to_csv(X, y, args.data_path)
