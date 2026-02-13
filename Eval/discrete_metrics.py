# Import methods 
import os
import sdv
import numpy as np
import pandas as pd
from scipy.special import kl_div
from sdv.metadata import Metadata
from scipy.spatial.distance import jensenshannon
from sdmetrics.single_column import KSComplement, TVComplement

# Variable declaration
# Paths for real and synthetic datasets
basePath = # INSERT PATH
filePath = # INSERT PATH

# Paths for saving split files - minority and majority - for real and synthetic datasets
splitRealPath = # INSERT PATH
splitSyntheticPath = # INSERT PATH

# Paths for saving evaluation results
save_path = # INSERT PATH

# List of datasets
datasets = ["adult", "australian", "GermanCredit_age25", "compas-scores", "ACSHI"]
generators = ["TabDDPM", "TabSyn", "CTABGAN"]
seeds = [42, 50, 61, 79, 83]

# Define columns for each dataset
dataset_columns = {
    "adult": {
        "continuous": ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week'],
        "categorical": ['education',  'marital-status', 'relationship', 'race', 'income'],
        "minority_column": 'gender',
        "minority_value": 'Female',
        "target_column": 'income-per-year'
    },
    "australian": {
        "continuous": ['A2', 'A3', 'A7', 'A10',  'A13', 'A14'],
        "categorical": ['A1', 'A4', 'A5', 'A6', 'A8', 'A9', 'A11', 'A12'],
        "minority_column": 'A15',
        "minority_value": 1,
        "target_column": 'A15'
    },
    "GermanCredit_age25": {
        "continuous": ['DurationMonth', 'CreditAmount', 'score'],
        "categorical": [],
        "minority_column": 'age25',
        "minority_value": 1,
        "target_column": 'credit'
    },
   "compas-scores": {
        "continuous": ['age',  'priors_count', 'juv_fel_count', 'juv_misd_count'],
        "categorical": ['sex', 'c_charge_degree', 'is_recid', 'is_violent_recid', 
			'v_decile_score', 'v_score_text', 'decile_score.1', 'score_text'],
        "minority_column": 'race',
        "minority_value": 'African-American',
        "target_column": 'two_year_recid'
    },
  "ACSHI": {
        "continuous": ['AGEP', 'PINCP'],
        "categorical": ['SCHL', 'ESP', 'CIT', 'MIG', 'MIL', 'ANC',
			'DEAR', 'DEYE', 'DREM', 'ESR', 'FER', 'label'],
        "minority_column": 'SEX',
        "minority_value": 1,
        "target_column": 'label'
    }
}

# Function to split data into majority and minority based on a specified column and value
def split_majority_minority(df, group_column, minority_value, file_prefix, filename):
    # Define the minority group
    minority_group = df[df[group_column] == minority_value]

    # Define the majority group (all rows that do not match the minority value)
    majority_group = df[df[group_column] != minority_value]

    # Save minority and majority groups as CSV files
    minority_file = os.path.join(filename, f"{file_prefix}_minority.csv")
    majority_file = os.path.join(filename, f"{file_prefix}_majority.csv")

    minority_group.to_csv(minority_file, index=False)
    majority_group.to_csv(majority_file, index=False)

    print(f"Minority group saved as {file_prefix}_minority.csv")
    print(f"Majority group saved as {file_prefix}_majority.csv")

    return minority_group, majority_group
	
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

# JS divergence for continuous and categorical columns
def calculate_js_divergence(original_df, synthetic_df, continuous_columns, categorical_columns, num_bins=30):
	# Create an empty dictionary to store the JS divergence scores
    js_divergence_scores = {}

    # Calculate JS divergence for continuous columns (using histograms)
    for column in continuous_columns:
        original_column = original_df[column].values
        synthetic_column = synthetic_df[column].values

        # Bin the data to create histograms (probability distributions)
        original_hist, _ = np.histogram(original_column, bins=num_bins, density=True)
        synthetic_hist, _ = np.histogram(synthetic_column, bins=num_bins, density=True)

        # Normalize the histograms
        original_dist = original_hist / original_hist.sum()
        synthetic_dist = synthetic_hist / synthetic_hist.sum()

        # Calculate JS divergence
        js_div_score = jensenshannon(original_dist, synthetic_dist)

        # Add the JS divergence score to the dictionary
        js_divergence_scores[column] = js_div_score

    # Calculate JS divergence for categorical columns (using frequency distributions)
    for column in categorical_columns:
        original_freq = original_df[column].value_counts(normalize=True).sort_index()
        synthetic_freq = synthetic_df[column].value_counts(normalize=True).sort_index()

        # Align the frequencies to ensure same categories
        all_categories = original_freq.index.union(synthetic_freq.index)
        original_dist = original_freq.reindex(all_categories, fill_value=0)
        synthetic_dist = synthetic_freq.reindex(all_categories, fill_value=0)

        # Calculate JS divergence
        js_div_score = jensenshannon(original_dist, synthetic_dist)

        # Add the JS divergence score to the dictionary
        js_divergence_scores[column] = js_div_score

    # Convert the dictionary to a DataFrame
    return pd.DataFrame(js_divergence_scores.items(), columns=['Column', 'JS Divergence Score'])

# KS complement for continuous columns
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

# TV complement for categorical columns
def calculate_tv_complement(original_df, synthetic_df, categorical_columns):
    # Create a dictionary to store the TV Complement scores
    tv_scores = {}

    for column in categorical_columns:
        try:
            score = TVComplement.compute(
                real_data=original_df[column],
                synthetic_data=synthetic_df[column]
            )
        except Exception as e:
            print(f"Error computing TV Complement for column '{column}': {e}")
            score = np.nan

        tv_scores[column] = score

    # Convert the dictionary to a DataFrame
    return pd.DataFrame({'Column': list(tv_scores.keys()), 'TV Complement Score': list(tv_scores.values())})

# Generate the evalution scores
# Main loop
for dataset in datasets:
    # Load real dataset
    real_df = pd.read_csv(f"{basePath}{dataset}.csv")
    
    columns = dataset_columns[dataset]
    continuous_columns = columns['continuous']
    categorical_columns = columns['categorical']
    minority_column = columns['minority_column']
    minority_value = columns['minority_value']
    
    # Split the real dataset into minority and majority groups
    minority_real_df, majority_real_df = split_majority_minority(real_df, minority_column, minority_value, f"real_{dataset}", splitRealPath)

    for generator in generators:
        for seed in seeds:
            # Load synthetic dataset
            synthetic_df = pd.read_csv(f"{filePath}{generator}_output_{dataset}_{seed}.csv")
            
            # Split the synthetic dataset into minority and majority groups
            minority_synthetic_df, majority_synthetic_df = split_majority_minority(synthetic_df, minority_column, minority_value, f"{generator}_{dataset}_{seed}", splitSyntheticPath)

            # 1. Evaluate for entire dataset
            kl_scores = calculate_kl_divergence(real_df, synthetic_df, continuous_columns)
            js_scores = calculate_js_divergence(real_df, synthetic_df, continuous_columns, categorical_columns)
            ks_scores = calculate_ks_complement(real_df, synthetic_df, continuous_columns)
            tv_scores = calculate_tv_complement(real_df, synthetic_df, categorical_columns)

            # Save evaluation results for entire dataset
            kl_scores.to_csv(f"{save_path}{generator}_{dataset}_{seed}_kl.csv", index=False)
            js_scores.to_csv(f"{save_path}{generator}_{dataset}_{seed}_js.csv", index=False)
            ks_scores.to_csv(f"{save_path}{generator}_{dataset}_{seed}_ks.csv", index=False)
            tv_scores.to_csv(f"{save_path}{generator}_{dataset}_{seed}_tv.csv", index=False)

            # 2. Evaluate for minority group
            kl_scores_minority = calculate_kl_divergence(minority_real_df, minority_synthetic_df, continuous_columns)
            js_scores_minority = calculate_js_divergence(minority_real_df, minority_synthetic_df, continuous_columns, categorical_columns)
            ks_scores_minority = calculate_ks_complement(minority_real_df, minority_synthetic_df, continuous_columns)
            tv_scores_minority = calculate_tv_complement(minority_real_df, minority_synthetic_df, categorical_columns)

            # Save evaluation results for minority group
            kl_scores_minority.to_csv(f"{save_path}{generator}_{dataset}_{seed}_minority_kl.csv", index=False)
            js_scores_minority.to_csv(f"{save_path}{generator}_{dataset}_{seed}_minority_js.csv", index=False)
            ks_scores_minority.to_csv(f"{save_path}{generator}_{dataset}_{seed}_minority_ks.csv", index=False)
            tv_scores_minority.to_csv(f"{save_path}{generator}_{dataset}_{seed}_minority_tv.csv", index=False)

            # 3. Evaluate for majority group
            kl_scores_majority = calculate_kl_divergence(majority_real_df, majority_synthetic_df, continuous_columns)
            js_scores_majority = calculate_js_divergence(majority_real_df, majority_synthetic_df, continuous_columns, categorical_columns)
            ks_scores_majority = calculate_ks_complement(majority_real_df, majority_synthetic_df, continuous_columns)
            tv_scores_majority = calculate_tv_complement(majority_real_df, majority_synthetic_df, categorical_columns)

            # Save evaluation results for majority group
            kl_scores_majority.to_csv(f"{save_path}{generator}_{dataset}_{seed}_majority_kl.csv", index=False)
            js_scores_majority.to_csv(f"{save_path}{generator}_{dataset}_{seed}_majority_js.csv", index=False)
            ks_scores_majority.to_csv(f"{save_path}{generator}_{dataset}_{seed}_majority_ks.csv", index=False)
            tv_scores_majority.to_csv(f"{save_path}{generator}_{dataset}_{seed}_majority_tv.csv", index=False)

print("Evaluation complete!")
