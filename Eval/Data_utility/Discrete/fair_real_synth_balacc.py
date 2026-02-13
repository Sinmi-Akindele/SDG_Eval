import os
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

# Paths for files
basePath = # INSERT PATH
filePath = # INSERT PATH

# Variable declaration
seeds = [42, 50, 61, 79, 83]

# Define the SDG models, datasets, and ML models with their metrics
synthetic_models = ["PATECTGAN", "CTGAN", "TVAE", "CTABGAN", "TabSyn", "TabDDPM"]  # Order matters to avoid partial matches
datasets = {"ACSHI", "adult", "australian", "compas-scores-two-years", "german_data"}

# Define ml models and metrics
models_and_metrics = {
    'logistic_regression': "balanced_accuracy_score",
    'xgboost': "balanced_accuracy_score",
    'mlp': "balanced_accuracy_score"
}

# Define dataset attributes
dataset_columns = {
    "ACSHI": {
        "continuous": ['AGEP', 'SCHL', 'ESP', 'CIT', 'MIG', 'MIL',
                      'ANC', 'DREM', 'PINCP', 'ESR'],
        "categorical": ['SEX', 'DEAR', 'DEYE', 'FER'],
        "minority_column": 'SEX',
        "minority_value": 1,
        "target_column": 'label'
    },
    "adult": {
        "continuous": ['age', 'education-num', 'capital-gain', 'capital-loss',
                       'hours-per-week'],
        "categorical": ['education', 'workclass', 'marital-status', 'occupation',
                        'relationship', 'race', 'sex', 'native-country'],
        "minority_column": 'sex',
        "minority_value": 'Female',
        "target_column": 'income-per-year'
    },
    "australian": {
        "continuous": ['A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A10', 'A12', 'A13', 'A14'],
        "categorical": ['A1', 'A8', 'A9', 'A11'],
        "minority_column": 'A15',
        "minority_value": 1,
        "target_column": 'A15'
    },
    "compas-scores-two-years": {
        "continuous": ['age',  'priors_count', 'juv_other_count'],
        "categorical": ['sex', 'age_cat', 'race', 'juv_fel_count', 'juv_misd_count',
                        'c_charge_degree', 'c_charge_desc'],
        "minority_column": 'sex',
        "minority_value": 'Female',
        "target_column": 'two_year_recid'
    },
    "german_data": {
        "continuous": ['month', 'credit_amount', 'investment_as_income_percentage',
                       'residence_since', 'age', 'number_of_credits', 'people_liable_for'],
        "categorical": ['status', 'credit_history', 'purpose', 'savings', 'employment', 'sex',
                        'other_debtors', 'property', 'installment_plans', 'housing',
                        'skill_level', 'telephone', 'foreign_worker'],
        "minority_column": 'sex',
        "minority_value": 'female',
        "target_column": 'credit'
    }
}

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
    if file_name.endswith(".csv"):
        # Print the current file being processed
        print(f"Processing file: {file_name}")

        # Parse dataset, SDG model, seed, ML model, and metric from filename
        parsed = parse_filename(file_name)

        # Skip files that don't match the required format
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

        # Get the minority column and separate based on minority and majority
        minority_value = dataset_columns[dataset_name]["minority_value"]
        minority_column = dataset_columns[dataset_name]["minority_column"]
        minority_mask = df[minority_column] == minority_value

        # Separate the minority and majority
        minority_df = df[minority_mask]
        majority_df = df[~minority_mask]

        # Calculate balanced accuracy for both minority and majority groups
        if metric == "balanced_accuracy_score":
            minority_score = balanced_accuracy_score(minority_df['y_true'], minority_df['y_pred'])
            majority_score = balanced_accuracy_score(majority_df['y_true'], majority_df['y_pred'])
        else:
            minority_score = None
            majority_score = None

        # Append to results for both minority and majority
        results.append({
            "Dataset": dataset_name,
            "SDG": sdg_model,
            "Seed": seed,
            "ML Model": model_name,
            "Minority": 1,
            "Evaluation Metric": metric,
            "Score": minority_score
        })

        results.append({
            "Dataset": dataset_name,
            "SDG": sdg_model,
            "Seed": seed,
            "ML Model": model_name,
            "Minority": 0,
            "Evaluation Metric": metric,
            "Score": majority_score
        })
    
# Convert results to DataFrame and save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(filePath, "balanced_accuracy_fairness_real_synthetic_type1_newSDG_results.csv"), index=False)

# Print the results to verify
print(results_df)
