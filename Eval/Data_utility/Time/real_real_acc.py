# Import methods
import os
import pandas as pd

# === Config ===
dataset_names = ["MetroTraffic", "Energy", "Tesla"]
model_types = ["gan"]
input_folder = # INSERT PATH
output_folder = # INSERT PATH
output_csv = os.path.join(output_folder, "compiled_eval_real_real_type2b_gan_results.csv")

# === Collect Results ===
results = []

for dataset in dataset_names:
    for model in model_types:
        filename = f"{dataset}_{model}_real_predictions.csv"
        filepath = os.path.join(input_folder, filename)

        # Check if file exists
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)

            # Make sure it has the right format (only 'mae' column)
            if "mae" in df.columns and len(df) == 1:
                score = df["mae"].iloc[0]
                results.append({
                    "Dataset": dataset,
                    "SDG": "real",
                    "ML Model": model,
                    "Test Type": "real",
                    "Evaluation Metric": "MAE",
                    "Score": score
                })
            else:
                print(f"Warning: Skipped malformed file {filename}")
        else:
            print(f"Warning: File not found {filename}")

# === Save to CSV ===
results_df = pd.DataFrame(results)
results_df.to_csv(output_csv, index=False)
print(f"Saved compiled results to: {output_csv}")
