import os
import pandas as pd

# === Config ===
dataset_names = ["MetroTraffic", "Energy", "Tesla"]
model_types = ["gan"]
synthetic_models = ["timegan", "tsdiff"]
trials = [0, 1, 2, 3, 4]

# File Path
input_folder = # INSERT PATH
output_folder = # INSERT PATH
output_csv = os.path.join(output_folder, "compiled_eval_real_synth_type2b_gan_results.csv")

# === Collect Results ===
results = []

for dataset in dataset_names:
    for model in model_types:
        for synth_model in synthetic_models:
            for trial in trials:
                filename = f"{synth_model}_{dataset}_{model}_predictions_trial_{trial}.csv"
                filepath = os.path.join(input_folder, filename)

                # Check if file exists
                if os.path.exists(filepath):
                    df = pd.read_csv(filepath)

                    # Validate and extract the MAE
                    if "mae" in df.columns and len(df) == 1:
                        score = df["mae"].iloc[0]
                        results.append({
                            "Dataset": dataset,
                            "SDG": synth_model,
                            "ML Model": model,
                            "Test Type": "synth",
                            "Evaluation Metric": "MAE",
                            "Score": score,
                            "Trial": trial
                        })
                    else:
                        print(f"Warning: Skipped malformed file {filename}")
                else:
                    print(f"Warning: File not found {filename}")

# === Save to CSV ===
results_df = pd.DataFrame(results)
results_df.to_csv(output_csv, index=False)
print(f"Saved compiled results to: {output_csv}")
