import os
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

# Paths for files
basePath = # INSERT PATH
filePath = # INSERT PATH

# Define the SDG models, datasets, and ML models with their metrics
synthetic_models = ["PATECTGAN", "CTGAN", "TVAE", "CTABGAN", "TabSyn", "TabDDPM"]  # Order matters to avoid partial matches
datasets = {"ACSHI", "adult", "australian", "compas-scores-two-years", "german_data"}

# Define ml models and metrics
models_and_metrics = {
    'logistic_regression': "balanced_accuracy_score",
    'xgboost': "balanced_accuracy_score",
    'mlp': "balanced_accuracy_score"
}

seeds = [42, 50, 61, 79, 83]

# Helper function to parse filename
def parse_filename(file_name):
    # Ensure the file matches the required format and ends with .csv
    if not file_name.endswith(".csv") or "_predictions_seed_" not in file_name:
        return None, None, None, None, None

    # Remove file extension
    base_name = file_name.replace(".csv", "")

    # Identify SDG model
    sdg_model = next((model for model in synthetic_models if model in base_name), None)
    if not sdg_model:
        return None, None, None, None, None

    # Identify dataset name
    dataset_name = next((dataset for dataset in datasets if dataset in base_name), None)
    if not dataset_name:
        return None, None, None, None, None

    # Identify ML model
    model_name = next((model for model in models_and_metrics if model in base_name), None)
    if not model_name:
        return None, None, None, None, None

    # Identify seed
    seed_str = base_name.split("_seed_")[-1]
    if seed_str.isdigit() and int(seed_str) in seeds:
        seed = int(seed_str)
    else:
        return None, None, None, None, None

    # Get evaluation metric based on ML model
    metric = models_and_metrics[model_name]

    return dataset_name, sdg_model, seed, model_name, metric

# Collect results in a list for easy conversion to DataFrame
results = []
for file_name in os.listdir(basePath):
    # Parse dataset, SDG model, seed, ML model, and metric from filename
    parsed = parse_filename(file_name)
    if parsed == (None, None, None, None, None):
        print(f"Skipping file: {file_name}")
        continue

    dataset_name, sdg_model, seed, model_name, metric = parsed

    # Read the predictions file
    file_path = os.path.join(basePath, file_name)
    df = pd.read_csv(file_path)

    # Check if columns 'y_true' and 'y_pred' exist in the file
    if 'y_true' not in df.columns or 'y_pred' not in df.columns:
        print(f"File missing required columns: {file_name}")
        continue

    # Calculate the score based on the metric
    if metric == "balanced_accuracy_score":
        score = balanced_accuracy_score(df['y_true'], df['y_pred'])
    else:
        score = None  # Placeholder in case an unknown metric is encountered

    # Append to results
    results.append({
        "Dataset": dataset_name,
        "SDG": sdg_model,
        "Seed": seed,
        "ML Model": model_name,
        "Evaluation Metric": metric,
        "Score": score
    })

# Convert results to DataFrame and save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(filePath, "balanced_accuracy_real_synthetic_type1_newSDG_results.csv"), index=False)

# Print the results to verify
print(results_df)
