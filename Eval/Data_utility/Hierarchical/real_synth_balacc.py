# Import methods
import os
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

# Paths for files
basePath = # INSERT PATH
filePath = # INSERT PATH

# Define the SDG models, datasets, and ML models with their metrics
synthetic_cases = [
    {"synthetic_model": "PAR", "dataset": "DIAG_wide"},
    {"synthetic_model": "CTGAN", "dataset": "DIAG_wide"},
    {"synthetic_model": "CTGAN", "dataset": "DIAG_numeric_decoding"},
    {"synthetic_model": "CTGAN", "dataset": "DIAG_auto_decoding"}
]

seeds = [42, 50, 61, 79, 83]
datasets = {"DIAG_wide", "DIAG_numeric_decoding", "DIAG_auto_decoding"}

# Define ml models and metrics
models_and_metrics = {
    'GAN': "balanced_accuracy_score",
    'LSTM': "balanced_accuracy_score",
    'RNN': "balanced_accuracy_score"
}

synthetic_models = {case["synthetic_model"] for case in synthetic_cases}

# Helper function to parse filename
def parse_filename(file_name):
    if not file_name.endswith(".csv"):
        return None, None, None, None, None

    base_name = file_name.replace(".csv", "")

    # Expecting format: {sdg_model}_output_{dataset}_{model_name}_predictions_seed_{seed}
    parts = base_name.split("_")
    try:
        # Find seed (last part)
        seed = int(parts[-1])
        if seed not in seeds:
            return None, None, None, None, None

        # Find 'predictions' keyword index
        predictions_idx = parts.index("predictions")

        # Extract model name (e.g., LSTM or GAN)
        model_name = parts[predictions_idx - 1]
        if model_name not in models_and_metrics:
            return None, None, None, None, None

        # Extract dataset name (assumes dataset name might have underscores)
        dataset_start_idx = 2  # fixed index after {sdg_model}_output_
        dataset_end_idx = predictions_idx - 2
        dataset_name = "_".join(parts[dataset_start_idx:dataset_end_idx + 1])
        if dataset_name not in datasets:
            return None, None, None, None, None

        # Extract sdg model
        sdg_model = parts[0]
        if sdg_model not in synthetic_models:
            return None, None, None, None, None

        # Metric from model
        metric = models_and_metrics[model_name]

        return dataset_name, sdg_model, seed, model_name, metric

    except (ValueError, IndexError):
        return None, None, None, None, None

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

    # Check if columns 'outputs_true' and 'outputs_pred' exist in the file
    if 'outputs_true' not in df.columns or 'outputs_pred' not in df.columns:
        print(f"File missing required columns: {file_name}")
        continue

    # Calculate the score based on the metric
    if metric == "balanced_accuracy_score":
        score = balanced_accuracy_score(df['outputs_true'], df['outputs_pred'])
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
results_df.to_csv(os.path.join(filePath, "balanced_accuracy_real_synthetic_type2a_DIAG__results.csv"), index=False)

# Print the results to verify
print(results_df)
