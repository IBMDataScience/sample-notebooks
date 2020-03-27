# Import modules.
import numpy as np
import pandas as pd

# Load data.
cars_data_y = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data', 
	names = ['price'], usecols = [25])

# Drop unknown values
cars_data_y = cars_data_y[cars_data_y.price != '?']

# Split the price column training and testing labels
from sklearn.model_selection import train_test_split
y_train, y_test = train_test_split(cars_data_y, test_size=0.2, random_state = 123)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.values.reshape(-1, 1))


import os

# Get input, output directories using environment variables
input_dir = os.environ.get('BATCH_INPUT_DIR')
output_dir = os.environ.get('BATCH_OUTPUT_DIR')

# Input, output, and log files
INPUT_FILE = os.path.join(input_dir, "car_results.csv")
OUTPUT_FILE = os.path.join(output_dir, "car_results_script_output.csv")
USER_LOG = os.path.join(output_dir, "user.log")

# Write to log file
def write_to_user_log(txt):
    with open(USER_LOG, 'a+') as f:
        f.write(txt + '\n')

if __name__ == '__main__':
    try:
        write_to_user_log("Start of script execution")

        # Read the input csv file downloaded from input_data_references
        results_df = pd.read_csv(INPUT_FILE)

        # Scale back predictions
        df_out = pd.DataFrame(sc_y.inverse_transform(results_df))

        # Write the processed dataframe to the output csv
        df_out.to_csv(OUTPUT_FILE, header=False, index=False)

        write_to_user_log("End of script execution")

    except Exception as ex:
        write_to_user_log('ERROR: ' + str(ex))

#sc_y.inverse_transform(np.array