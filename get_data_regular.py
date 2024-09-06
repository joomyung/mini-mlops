import numpy as np
import csv
import argparse
import time
import os

def check_most_recent_data(data_folder_path):
    """Return index for train."""
    data = os.listdir(data_folder_path)
    # sort the data by index
    data.sort(key=lambda x: int(x.split(".")[0].split("_")[1]))
    recent_data = data[-1]
    recent_data_index = recent_data.split(".")[0].split("_")[1]
    index_to_gen_data = int(recent_data_index) + 1
    return index_to_gen_data

def get_data_regular_interval():
    """Generate linear data with noise."""
    X = np.linspace(0, 10, 100)
    true_slope = 2.5
    true_intercept = -1.0
    noise = np.random.randn(100) * 2
    y = true_slope * X + true_intercept + noise
    return X, y

def save_data_to_csv_regular_interval(X, y, data_folder_path, index_to_gen_data):
    """Save data to CSV.""" 
    with open(f"{data_folder_path}/data_{index_to_gen_data}.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['X', 'y'])
        for x_val, y_val in zip(X, y):
            writer.writerow([x_val, y_val])
    #print("Data saved to", data_path)

def parse_arguments():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description='Generate linear data with noise at regular interval.')
    parser.add_argument('data_folder_path', type=str, help='Output data CSV file path.')
    parser.add_argument('interval', type=str, help='Regular interval to generate data.')
    return parser.parse_args()

if __name__ == "__main__":
    # python get_data_regular.py data 5
    args = parse_arguments()
    index_to_gen_data = check_most_recent_data(args.data_folder_path)
    while True:
        try:
            X, y = get_data_regular_interval()
            save_data_to_csv_regular_interval(X, y, args.data_folder_path, index_to_gen_data)
            index_to_gen_data += 1
            time.sleep(int(args.interval))
        except Exception as e:
            print(e)
            break
