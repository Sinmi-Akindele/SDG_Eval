# Import methods
import os
import pandas as pd
from sklearn.metrics import accuracy_score

# Paths for files
basePath = # INSERT PATH
filePath = # INSERT PATH

# Define dataset attributes
dataset_columns = {
    "PIAAC_wide": {
        "minority_column": 'GENDER_R',
        "minority_value": 'Male'
    }
}

# Variable declaration
datasets = {"PIAAC_wide"}

# Define ml models and metrics
models_and_metrics = {
    'GAN': "accuracy",
    'LSTM': "accuracy",
    'RNN': "accuracy"
}

# Helper function to parse filename based on known dataset names and format
def parse_filename(file_name):
    # Ensure the file matches the required format
    if not file_name.endswith("_real_predictions.csv"):
        return None, None, None

    # Remove file extension
    base_name = file_name.replace(".csv", "")

    # Identify dataset name
    dataset_name = next((dataset for dataset in datasets if dataset in base_name), None)
    if not dataset_name:
        return None, None, None

    # Check remaining part for model name and real_predictions
    remaining_parts = base_name.replace(dataset_name + "_", "")
    model_name = next((model for model in models_and_metrics if model in remaining_parts), None)

    # Confirm presence of model name and "real_predictions" suffix
    if model_name and "real_predictions" in remaining_parts:
        metric = models_and_metrics[model_name]
        return dataset_name, model_name, metric

    return None, None, None

# Collect results in a list for easy conversion to DataFrame
results = []
for file_name in os.listdir(basePath):
    if file_name.endswith(".csv"):
        # Print the current file being processed
        print(f"Processing file: {file_name}")

        # Parse dataset, model, and metric from filename
        dataset_name, model_name, metric = parse_filename(file_name)

        # Skip files that don't match the required format
        if not dataset_name or not model_name:
            continue

        file_path = os.path.join(basePath, file_name)

        # Read the predictions file
        df = pd.read_csv(file_path)

        # Get the minority column and separate based on minority and majority
        minority_value = dataset_columns[dataset_name]["minority_value"]
        minority_column = dataset_columns[dataset_name]["minority_column"]
        minority_mask = df[minority_column] == minority_value

        # Separate the minority and majority
        minority_df = df[minority_mask]
        majority_df = df[~minority_mask]

        # Calculate accuracy for both minority and majority groups
        if metric == "accuracy":
            minority_score = accuracy_score(minority_df['outputs_true'], minority_df['outputs_pred'])
            majority_score = accuracy_score(majority_df['outputs_true'], majority_df['outputs_pred'])
        else:
            minority_score = None
            majority_score = None

        # Append to results for both minority and majority
        results.append({
            "Dataset": dataset_name,
            "ML Model": model_name,
            "Minority": 1,
            "Evaluation Metric": metric,
            "Score": minority_score
        })

        results.append({
            "Dataset": dataset_name,
            "ML Model": model_name,
            "Minority": 0,
            "Evaluation Metric": metric,
            "Score": majority_score
        })

# Convert results to DataFrame and save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(filePath, "compiled_fairness_real_real_type2a_PIAAC_results.csv"), index=False)

# Print the results to verify
print(results_df)
