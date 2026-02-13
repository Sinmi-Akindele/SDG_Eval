# Import methods
import os
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_absolute_error
from tf_slim.layers import layers as _layers
from sklearn.model_selection import train_test_split

# Enable eager execution
tf.compat.v1.enable_eager_execution()

# === Config ===
synthetic_models = ["timegan", "tsdiff"]
dataset_names = ["MetroTraffic", "Energy", "Tesla"]
trials = [0, 1, 2, 3, 4]
model_types = ["rnn", "lstm"]
seq_len = 24

# Paths
basePath = # INSERT PATH
synthPath = # INSERT PATH
filePath = # INSERT PATH

# Helper Functions
# Normalize both datasets (min-max scaling between 0 and 1)
def min_max_normalize(df):
    return (df - df.min()) / (df.max() - df.min() + 1e-7)

def make_sequences(data, seq_len):
    sequences = []
    for i in range(0, len(data) - seq_len + 1):
        seq = data[i:i + seq_len].to_numpy()
        sequences.append(seq)
    return np.array(sequences)

def predictive_score(model_type, ori_data, generated_data, test_type="real"):
    generated_data = np.array(generated_data)
    no, seq_len, dim = generated_data.shape
    hidden_dim = int(dim / 2)
    iterations = 30
    activation = tf.nn.tanh

    if test_type == "real":
        ori_data = np.array(ori_data)
        X_train = generated_data[:, :-1, :]
        Y_train = generated_data[:, 1:, :]
        X_test = ori_data[:, :-1, :]
        Y_test = ori_data[:, 1:, :]

    elif test_type == "synth":
        train_idx, test_idx = train_test_split(np.arange(len(generated_data)), test_size=0.2, random_state=42)
        X_train = generated_data[train_idx, :-1, :]
        Y_train = generated_data[train_idx, 1:, :]
        X_test = generated_data[test_idx, :-1, :]
        Y_test = generated_data[test_idx, 1:, :]
    else:
        raise ValueError("Unknown test_type: choose 'real' or 'synth'")

    # Define model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(seq_len - 1, dim)))
    if model_type == "rnn":
        model.add(tf.keras.layers.GRU(units=hidden_dim, activation=activation, return_sequences=True))
    elif model_type == "lstm":
        model.add(tf.keras.layers.LSTM(units=hidden_dim, activation=activation, return_sequences=True))
    else:
        raise ValueError("Unknown model type")

    model.add(tf.keras.layers.Dense(units=dim, activation='sigmoid'))
    model.compile(optimizer='adam', loss='mae')

    # Train on synthetic data
    model.fit(X_train, Y_train, batch_size=128, epochs=iterations, verbose=0)

    # Predict
    Y_pred = model.predict(X_test, verbose=0)

    # MAE per sequence
    mae_list = [mean_absolute_error(Y_test[i], Y_pred[i]) for i in range(len(Y_test))]
    return np.mean(mae_list)

# === Loop Through All Combinations ===
results = []

# === Train on real, test on real and synthetic ===
for dataset in dataset_names:
    # Load and preprocess real data
    real_data_path = os.path.join(basePath, f"{dataset}.csv")
    real_df = pd.read_csv(real_data_path)
    if dataset == "MetroTraffic":
        real_df = real_df.drop(columns=['holiday', 'weather_main', 'weather_description', 'date_time'])
    elif dataset == "Energy":
        real_df = real_df.drop(columns=['date'])
    elif dataset == "Tesla":
        real_df = real_df.drop(columns=['Id', 'Date'])
    real_df = min_max_normalize(real_df)
    full_real_seqs = make_sequences(real_df, seq_len)

    # Split once for consistency
    train_idx, test_idx = train_test_split(np.arange(len(full_real_seqs)), test_size=0.2, random_state=42)
    real_train = full_real_seqs[train_idx]
    real_test = full_real_seqs[test_idx]

    for model_name in model_types:
        # === Train on real, test on held-out real ===
        mae_real = predictive_score(model_name, real_test, real_train, test_type="real")
        results.append({
            "Dataset": dataset,
            "SDG": "real",
            "ML Model": model_name,
            "Test Type": "real",
            "Evaluation Metric": 'MAE',
            "Score": mae_real
        })

        real_real_path = os.path.join(filePath, f"{dataset}_{model_name}_real_predictions.csv")
        pd.DataFrame([{"mae": mae_real}]).to_csv(real_real_path, index=False)
        print(f"Saved: {real_real_path}")

        for synth_model in synthetic_models:
            for trial in trials:
                synthetic_path = os.path.join(synthPath, f"{synth_model}_output_{dataset}_trial_{trial}.csv")
                if not os.path.exists(synthetic_path):
                    print(f"Missing file: {synthetic_path}")
                    continue
                synthetic_df = pd.read_csv(synthetic_path)
                synthetic_df = min_max_normalize(synthetic_df)
                generated_data = make_sequences(synthetic_df, seq_len)

                # === Train on real, test on synthetic ===
                mae_synth = predictive_score(model_name, generated_data, real_train, test_type="synth")
                results.append({
                    "Dataset": dataset,
                    "SDG": synth_model,
                    "Trial": trial,
                    "ML Model": model_name,
                    "Test Type": "synth",
                    "Evaluation Metric": 'MAE',
                    "Score": mae_synth
                })

                real_synth_path = os.path.join(
                    filePath, f"{synth_model}_output_{dataset}_{model_name}_predictions_trial_{trial}.csv"
                )
                pd.DataFrame([{"mae": mae_synth}]).to_csv(real_synth_path, index=False)
                print(f"Saved: {real_synth_path}")
