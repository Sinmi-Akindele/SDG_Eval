# Import methods
import os
import numpy as np
import pandas as pd
from scipy.special import kl_div
from sdmetrics.single_column import KSComplement
from sklearn.metrics import r2_score as sklearn_r2

# Variable declaration
# Paths for real and synthetic datasets
basePath = # INSERT PATH
filePath = # INSERT PATH

# Paths for saving evaluation results
save_path = # INSERT PATH

# List of datasets
datasets = ["MetroTraffic","Energy","Tesla"]
generators = ["timegan", "tsdiff"]
trials = [0, 1, 2, 3, 4]

# Define columns for each dataset
dataset_columns = {
    "MetroTraffic":{
        "continuous": ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'traffic_volume']
    },
    "Energy": {
        "continuous": ['Appliances', 'lights', 'T1', 'RH_1', 'T2', 'RH_2', 'T3', 
                       'RH_3', 'T4', 'RH_4', 'T5', 'RH_5', 'T6', 'RH_6', 'T7',
                       'RH_7', 'T8', 'RH_8', 'T9', 'RH_9', 'T_out', 'Press_mm_hg', 
                       'RH_out', 'Windspeed', 'Visibility', 'Tdewpoint', 'rv1', 'rv2']
    },
    "Tesla": {
        "continuous": ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    }
}

# KL divergence for continuous columns, adjusting for unequal lengths only when necessary
def calculate_kl_divergence(original_df, synthetic_df, continuous_columns, epsilon=1e-12):
    # Create an empty dictionary to store the KL divergence scores
    kl_divergence_scores = {}

    # Iterate over each continuous column
    for column in continuous_columns:
        # Extract the column data from both datasets
        original_column = original_df[column].values
        synthetic_column = synthetic_df[column].values

        # Check if the lengths of the columns are unequal
        if len(original_column) != len(synthetic_column):
            # Adjust for unequal lengths by taking the shorter length of the two
            min_len = min(len(original_column), len(synthetic_column))
            original_column = original_column[:min_len]
            synthetic_column = synthetic_column[:min_len]

        # Normalize the columns to ensure they represent a probability distribution
        original_column = original_column / np.sum(original_column)
        synthetic_column = synthetic_column / np.sum(synthetic_column)

        # Add epsilon to prevent division by zero or taking log of zero
        original_column = np.clip(original_column, epsilon, None)
        synthetic_column = np.clip(synthetic_column, epsilon, None)

        # Calculate KL divergence using scipy's kl_div function
        kl_div_score = np.sum(kl_div(original_column, synthetic_column))

        # Add the KL divergence score to the dictionary
        kl_divergence_scores[column] = kl_div_score

    # Convert the dictionary to a DataFrame for easy interpretation
    return pd.DataFrame({'Column': list(kl_divergence_scores.keys()), 'KL Divergence Score': list(kl_divergence_scores.values())})

# R2 score for continuous columns
def calculate_r2_score(original_df, synthetic_df, continuous_columns):
    # Create an empty dictionary to store the R2 scores
    R2_scores = {}
    
    # Iterate over each continuous column
    for column in continuous_columns:
      # Extract the column data from both datasets
      original_column = original_df[column].values
      synthetic_column = synthetic_df[column].values
      
      # Check if the lengths of the columns are unequal
      if len(original_column) != len(synthetic_column):
        # Adjust for unequal lengths by taking the shorter length of the two
        min_len = min(len(original_column), len(synthetic_column))
        original_column = original_column[:min_len]
        synthetic_column = synthetic_column[:min_len]

      # Calculate R2 score using sklearn's r2_score function
      r2_score = sklearn_r2(original_column, synthetic_column)
      
      # Add the R2 score to the dictionary
      R2_scores[column] = r2_score

    # Convert the dictionary to a DataFrame for easy interpretation
    return pd.DataFrame({'Column': list(R2_scores.keys()), 'R2 Score': list(R2_scores.values())})

# KS Complement for continuous columns
def calculate_ks_complement(original_df, synthetic_df, continuous_columns):
    # Create a dictionary to store the KS Complement scores
    ks_scores = {}

    for column in continuous_columns:
        try:
            score = KSComplement.compute(
                real_data=original_df[column],
                synthetic_data=synthetic_df[column]
            )
        except Exception as e:
            print(f"Error computing KS Complement for column '{column}': {e}")
            score = np.nan

        ks_scores[column] = score

    # Convert the dictionary to a DataFrame
    return pd.DataFrame({'Column': list(ks_scores.keys()), 'KS Complement Score': list(ks_scores.values())})

# Generate the evaluation scores
# Main loop
for dataset in datasets:
    # Load real dataset
    real_df = pd.read_csv(f"{basePath}{dataset}.csv")

    columns = dataset_columns[dataset]
    continuous_columns = columns['continuous']

    for generator in generators:
        for trial in trials:
            # Load synthetic dataset
            synthetic_df = pd.read_csv(f"{filePath}{generator}_output_{dataset}_trial_{trial}.csv")

            # 1. Evaluate for entire dataset
            kl_scores = calculate_kl_divergence(real_df, synthetic_df, continuous_columns)
            r2_scores = calculate_r2_score(real_df, synthetic_df, continuous_columns)
            ks_scores = calculate_ks_complement(real_df, synthetic_df, continuous_columns)

            # Save evaluation results for entire dataset
            kl_scores.to_csv(f"{save_path}{generator}_{dataset}_{trial}_kl.csv", index=False)
            r2_scores.to_csv(f"{save_path}{generator}_{dataset}_{trial}_r2.csv", index=False)
            ks_scores.to_csv(f"{save_path}{generator}_{dataset}_{trial}_ks.csv", index=False)

print("Evaluation complete!")
