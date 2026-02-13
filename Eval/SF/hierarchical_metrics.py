# Import methods
import os
import glob
import numpy as np
import pandas as pd
from collections import Counter
from sdmetrics.single_column.statistical import TVComplement

# Variable declaration
seeds = [42, 50, 61, 79, 83]
group_column = "ETHNICITY"
minority_value = "Hispanic Origin"
dataset_name = ["PROC_wide", "PROC_numeric_encoding", "PROC_auto_encoding"]

# File path
realPath = # INSERT PATH
synthPath = # INSERT PATH

# Paths for saving split files - minority and majority - for real and synthetic datasets
splitRealPath = # INSERT PATH
splitSyntheticPath = # INSERT PATH

# Paths for saving evaluation results
savePath = # INSERT PATH

# ________________Function declartion________________
# ________________Split dataset to minority and majority________________
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

# ROUGE-1
# Define the function to calculate ROUGE-1 score for a single pair of sequences
def calculate_rouge_1(real_seq, synthetic_seq):
    # Tokenize the sequences into unigrams (words)
    real_tokens = real_seq.split(",")
    synthetic_tokens = synthetic_seq.split(",")

    # Count the unigrams in both the real and synthetic sentences
    real_counter = Counter(real_tokens)
    synthetic_counter = Counter(synthetic_tokens)

    # Calculate the number of unigrams in the synthetic sentence that appear in the real sentence
    common_unigrams = sum((real_counter & synthetic_counter).values())

    # Calculate ROUGE-1 score
    rouge_1 = common_unigrams / len(synthetic_tokens)

    return rouge_1

# Define the function to calculate average ROUGE-1 score for a column
def calculate_avg_rouge_1(real_df, synthetic_df, dataset_name, column_name):
    # Determine the column name based on dataset_name
    if dataset_name == "PROC_auto_encoding":
        column_name = "PS_Auto_Encoding"
    elif dataset_name == "PROC_numeric_encoding":
        column_name = "PS_Numeric_Encoding"
    elif dataset_name == "PROC_wide":
        column_name = "ALL_PROC_CODES"
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    # Get the relevant column for comparison
    real_column = real_df[column_name]
    synthetic_column = synthetic_df[column_name]
    print(real_column)
    print(synthetic_column)

    # Calculate ROUGE-1 score for each row
    rouge_1_scores = [
        calculate_rouge_1(real, synthetic) for real, synthetic in zip(real_column, synthetic_column)
    ]

    # Calculate the average ROUGE-1 score
    avg_rouge_1 = sum(rouge_1_scores) / len(rouge_1_scores) if rouge_1_scores else 0

    return avg_rouge_1

# ROUGE-L
# Function to calculate length of Longest Common Subsequence (LCS)
def calculate_lcs(real_seq, synthetic_seq):
    # Tokenize the sequences into words
    real_tokens = real_seq.split(",")
    synthetic_tokens = synthetic_seq.split(",")

    # Create a 2D table to store the LCS lengths
    m, n = len(real_tokens), len(synthetic_tokens)
    lcs_table = [[0] * (n + 1) for _ in range(m + 1)]

    # Fill the table using dynamic programming
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if real_tokens[i - 1] == synthetic_tokens[j - 1]:
                lcs_table[i][j] = lcs_table[i - 1][j - 1] + 1
            else:
                lcs_table[i][j] = max(lcs_table[i - 1][j], lcs_table[i][j - 1])

    # The LCS length is in the bottom-right cell of the table
    lcs_length = lcs_table[m][n]

    return lcs_length

# Function to calculate ROUGE-L score for a single pair of sequences
def calculate_rouge_L(real_seq, synthetic_seq):
    # Get the length of the longest common subsequence
    lcs_length = calculate_lcs(real_seq, synthetic_seq)

    # Tokenize the sequences into words
    real_tokens = real_seq.split(",")
    synthetic_tokens = synthetic_seq.split(",")

    # Calculate ROUGE-L score
    rouge_L = lcs_length / len(synthetic_tokens)

    return rouge_L

# Function to calculate average ROUGE-L score for a column
def calculate_avg_rouge_L(real_df, synthetic_df, dataset_name, column_name):
    # Determine the column name based on dataset_name
    if dataset_name == "PROC_auto_encoding":
        column_name = "PS_Auto_Encoding"
    elif dataset_name == "PROC_numeric_encoding":
        column_name = "PS_Numeric_Encoding"
    elif dataset_name == "PROC_wide":
        column_name = "ALL_PROC_CODES"
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    # Get the relevant column for comparison
    real_column = real_df[column_name]
    synthetic_column = synthetic_df[column_name]
    print(real_column)
    print(synthetic_column)

    # Calculate ROUGE-L score for each row
    rouge_L_scores = [
        calculate_rouge_L(real, synthetic) for real, synthetic in zip(real_column, synthetic_column)
    ]

    # Calculate the average ROUGE-L score
    avg_rouge_L = sum(rouge_L_scores) / len(rouge_L_scores) if rouge_L_scores else 0

    return avg_rouge_L

# TV complement for categorical columns
def calculate_tv_complement(real_df, synthetic_df, dataset_name, column_name):
    # Determine the column name based on dataset_name
    if dataset_name == "PROC_auto_encoding":
        column_name = "PS_Auto_Encoding"
    elif dataset_name == "PROC_numeric_encoding":
        column_name = "PS_Numeric_Encoding"
    elif dataset_name == "PROC_wide":
        column_name = "ALL_PROC_CODES"
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    try:
        score = TVComplement.compute(
            real_data=real_df[column_name],
            synthetic_data=synthetic_df[column_name]
        )
    except Exception as e:
        print(f"Error computing TV Complement for column '{column_name}': {e}")
        score = np.nan

    return score

def save_score(score, filename):
    with open(filename, 'w') as f:
        f.write(f"{score:.4f}\n")

# Main Loop
for dataset in dataset_name:
  # Load real dataset
  real_df = pd.read_csv(f"{realPath}{dataset}.csv")

  # Split the real dataset into minority and majority groups
  min_real_df, maj_real_df = split_majority_minority(real_df, group_column, minority_value, f"{dataset}", splitRealPath)

  for seed in seeds:
        # Define synthetic file paths based on dataset
        if dataset in ["PROC_auto_encoding", "PROC_numeric_encoding"]:
            # Load synthetic dataset
            synth_file = f"CTGAN_output_{dataset}_{seed}.csv"
            synth_path_full = os.path.join(synthPath, synth_file)
            
            # Check if file exists
            if not os.path.exists(synth_path_full):
              print(f"File not found: {synth_path_full}. Skipping...")
              continue

            # Load synthetic dataset
            synth_df = pd.read_csv(os.path.join(synth_path_full))

            # Split the synthetic dataset into minority and majority groups
            min_synth_df, maj_synth_df = split_majority_minority(synth_df, group_column, minority_value, f"CTGAN_{dataset}_{seed}", splitSyntheticPath)

            # 1. Evaluate for entire dataset
            r1 = calculate_avg_rouge_1(real_df, synth_df, dataset, None)
            rl = calculate_avg_rouge_L(real_df, synth_df, dataset, None)
            tv = calculate_tv_complement(real_df, synth_df, dataset, None)

            # Save evaluation results for entire dataset
            save_score(r1, f"{savePath}{synth_file}_Rouge_1.csv")
            save_score(rl, f"{savePath}{synth_file}_Rouge_L.csv")
            save_score(tv, f"{savePath}{synth_file}_TV_Complement.csv")

            # 2. Evaluate for minority group
            r1_min = calculate_avg_rouge_1(min_real_df, min_synth_df, dataset, None)
            rl_min = calculate_avg_rouge_L(min_real_df, min_synth_df, dataset, None)
            tv_min = calculate_tv_complement(min_real_df, min_synth_df, dataset, None)

            # Save evaluation results for minority group
            save_score(r1_min, f"{savePath}{synth_file}_minority_Rouge_1.csv")
            save_score(rl_min, f"{savePath}{synth_file}_minority_Rouge_L.csv")
            save_score(tv_min, f"{savePath}{synth_file}_minority_TV_Complement.csv")

            # 3. Evaluate for majority group
            r1_maj = calculate_avg_rouge_1(maj_real_df, maj_synth_df, dataset, None)
            rl_maj = calculate_avg_rouge_L(maj_real_df, maj_synth_df, dataset, None)
            tv_maj = calculate_tv_complement(maj_real_df, maj_synth_df, dataset, None)

            # Save evaluation results for majority group
            save_score(r1_maj, f"{savePath}{synth_file}_majority_Rouge_1.csv")
            save_score(rl_maj, f"{savePath}{synth_file}_majority_Rouge_L.csv")
            save_score(tv_maj, f"{savePath}{synth_file}_majority_TV_Complement.csv")

        elif dataset == "PROC_wide":
            for SDG in ["CTGAN", "PAR"]:
              # Construct file path
              synth_file = f"{SDG}_output_{dataset}_{seed}.csv"
              synth_path_full = os.path.join(synthPath, synth_file)

              # Check if file exists
              if not os.path.exists(synth_path_full):
                print(f"File not found: {synth_path_full}. Skipping...")
                continue

              # Load synthetic dataset
              synth_df = pd.read_csv(synth_path_full)

              # Split the synthetic dataset into minority and majority groups
              min_synth_df, maj_synth_df = split_majority_minority(synth_df, group_column, minority_value, f"{SDG}_{dataset}_{seed}", splitSyntheticPath)

              # 1. Evaluate for entire dataset
              r1 = calculate_avg_rouge_1(real_df, synth_df, dataset, None)
              rl = calculate_avg_rouge_L(real_df, synth_df, dataset, None)
              tv = calculate_tv_complement(real_df, synth_df, dataset, None)

              # Save evaluation results for entire dataset
              save_score(r1, f"{savePath}{synth_file}_Rouge_1.csv")
              save_score(rl, f"{savePath}{synth_file}_Rouge_L.csv")
              save_score(tv, f"{savePath}{synth_file}_TV_Complement.csv")

              # 2. Evaluate for minority group
              r1_min = calculate_avg_rouge_1(min_real_df, min_synth_df, dataset, None)
              rl_min = calculate_avg_rouge_L(min_real_df, min_synth_df, dataset, None)
              tv_min = calculate_tv_complement(min_real_df, min_synth_df, dataset, None)

              # Save evaluation results for minority group
              save_score(r1_min, f"{savePath}{synth_file}_minority_Rouge_1.csv")
              save_score(rl_min, f"{savePath}{synth_file}_minority_Rouge_L.csv")
              save_score(tv_min, f"{savePath}{synth_file}_minority_TV_Complement.csv")

              # 3. Evaluate for majority group
              r1_maj = calculate_avg_rouge_1(maj_real_df, maj_synth_df, dataset, None)
              rl_maj = calculate_avg_rouge_L(maj_real_df, maj_synth_df, dataset, None)
              tv_maj = calculate_tv_complement(maj_real_df, maj_synth_df, dataset, None)

              # Save evaluation results for majority group
              save_score(r1_maj, f"{savePath}{synth_file}_majority_Rouge_1.csv")
              save_score(rl_maj, f"{savePath}{synth_file}_majority_Rouge_L.csv")
              save_score(tv_maj, f"{savePath}{synth_file}_majority_TV_Complement.csv")
