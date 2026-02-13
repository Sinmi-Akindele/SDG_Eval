# Import methods
import os
import ast
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from collections import Counter
from sklearn.metrics import accuracy_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.models import Sequential
from torch.utils.data import Dataset, DataLoader
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import GaussianNoise
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, Dense, Dropout


# Set the base path and parameters
basePath = # INSERT PATH
real_dataset_path = # INSERT PATH
real_action_column = "ALL_DIAG_CODES"
output_path = os.path.join(basePath, "MlEfficacy2A")

synthetic_cases = [
    {"folder": os.path.join(basePath, "Synthetic"), "model": "PAR", "dataset": "DIAG_wide", "column": "ALL_DIAG_CODES"},
    {"folder": os.path.join(basePath, "Synthetic"), "model": "CTGAN", "dataset": "DIAG_wide", "column": "ALL_DIAG_CODES"},
    {"folder": os.path.join(basePath, "Synthetic"), "model": "CTGAN", "dataset": "DIAG_numeric_decoding", "column": "DS_Numeric_Decoding"},
    {"folder": os.path.join(basePath, "Synthetic"), "model": "CTGAN", "dataset": "DIAG_auto_decoding", "column": "DS_Auto_Decoding"}
]

# Variable declaration
seeds = [42, 50, 61, 79, 83]

# Hyper-parameters
input_length=15
embedding_dim=128
rnn_units=128
dropout_rate=0.1
learning_rate=0.001
batch_size=64
epochs=150

###############################
# A. Data Preprocessing Function
###############################
def preprocess_data(data, action_column, input_length):
    """
    Preprocess diagnosis sequences where:
      - target = most recent diagnosis (first token)
      - input = remaining diagnoses sorted by global frequency (most → least)
    """
    # 1. Load data
    df = pd.read_csv(data)
    
    # Convert comma-separated diagnoses to space-separated (for Keras tokenizer)
    df[action_column] = df[action_column].astype(str).apply(lambda x: " ".join(x.split(",")))
    
    # Keep only relevant columns
    df = df[["SEX_CODE", action_column]]

    # 2. Build global frequency table
    all_diags = [d for seq in df[action_column] for d in seq]
    diag_freq = Counter(all_diags)
    
    # 3. Tokenization
    tokenizer = Tokenizer(lower=False, filters='')
    tokenizer.fit_on_texts([" ".join(seq) for seq in df[action_column]])
    vocab_size = len(tokenizer.word_index) + 1  # +1 for padding token
    
    # 4. Convert sequences to token IDs
    df["TokenizedSequence"] = tokenizer.texts_to_sequences(
        [" ".join(seq) for seq in df[action_column]]
    )
    
    # Build inverse map for frequency lookup
    idx2diag = {v: k for k, v in tokenizer.word_index.items()}

    # 5. Build input-output pairs
    inputs, outputs, genders = [], [], []
    
    for idx, diag_seq in enumerate(df["TokenizedSequence"]):
        if len(diag_seq) < 2:
            continue  # skip sequences too short to predict first diagnosis
        
        target = diag_seq[0]            # first diagnosis = target
        input_seq = diag_seq[1:]       # remaining diagnoses = input

        # Sort input diagnoses by global frequency (descending)
        input_seq = sorted(
            input_seq,
            key=lambda x: diag_freq[idx2diag[x]],
            reverse=True
        )
        
        # Truncate/pad input sequence
        input_seq = input_seq[:input_length]
        pad_len = input_length - len(input_seq)
        input_seq = input_seq + [0] * pad_len  # 0 is padding token
        
        inputs.append(input_seq)
        outputs.append(target)

        # Record corresponding SEX_CODE
        genders.append(df["SEX_CODE"].iloc[idx])
    
    # 6. Preprocess inputs and outputs
    inputs = np.array(inputs)
    outputs = to_categorical(outputs, num_classes=vocab_size)
    
    return inputs, outputs, genders, tokenizer, vocab_size

###############################
# B. Model Building Function
###############################
def build_rnn_model(vocab_size, input_length, embedding_dim, rnn_units, dropout_rate, learning_rate):
    # Build the RNN model
    model = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=input_length,
            mask_zero=True
        ),
        GaussianNoise(0.3),
        SimpleRNN(rnn_units),
        Dropout(dropout_rate),
        Dense(vocab_size, activation='softmax')
    ])

    # Compile the model
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate),
        metrics=['accuracy']
    )

    return model

###############################
# C. Evaluation & Prediction Saving Function
###############################
def evaluate_and_save_predictions(model, X_test, y_test, gender_test, output_file):
    """
    Generates predictions on the provided X, compares with true labels y, and saves a CSV file with:
      - GENDER_R (if available)
      - outputs_true: the true token IDs
      - outputs_pred: the predicted token IDs
    """
    # === 11. Save real data predictions to CSV ===
    # Generate prediction probabilities and get predicted token IDs
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Create a DataFrame for predictions; include GENDER_R
    pred_df = pd.DataFrame({
            "SEX_CODE": gender_test,
            "outputs_true": y_true,
            "outputs_pred": y_pred
    })

    # Save the predictions to a CSV file
    pred_df.to_csv(output_file, index=False)
    print(f"Saved predictions to {output_file}")

###############################
# D. Function to Run Predictions on Synthetic Files
###############################
def predict_on_synthetic(model, tokenizer, synth_data, action_column, output_file, input_length):
    """
    Loads a synthetic dataset, processes the action sequence column, creates input sequences (first 4 tokens),
    uses the trained model to predict the next action, and saves the predictions.

    The output CSV will have:
      - GENDER_R (if available)
      - outputs_true (if applicable, i.e. the 5th token if present)
      - outputs_pred (predicted next token)
    """

    # 1. Load synthetic data
    df = pd.read_csv(synth_data)

    # Convert the action sequence from comma-separated to space-separated
    df[action_column] = df[action_column].astype(str).apply(lambda x: " ".join(x.split(",")))

    # Keep only relevant columns
    df_subset = df[["SEX_CODE", action_column]]

    # 2. Build global frequency table FROM SYNTHETIC DATA
    all_diags = [d for seq in df_subset[action_column] for d in seq]
    diag_freq = Counter(all_diags)

    # 3. Tokenization
    # Tokenize using the existing tokenizer
    df_subset["TokenizedSequence"] = tokenizer.texts_to_sequences(
        [" ".join(seq) for seq in df_subset[action_column]]
    )

    # Inverse tokenizer mapping for frequency lookup
    idx2diag = {v: k for k, v in tokenizer.word_index.items()}

    # 4. Build input-output pairs
    inputs, outputs, genders = [], [], []

    # Create input-output pairs from the synthetic data, if possible.
    for idx, diag_seq in enumerate(df_subset["TokenizedSequence"]):
        if len(diag_seq) < 2:
            continue  # skip sequences too short to predict first diagnosis

        target = diag_seq[0]            # first diagnosis = target
        input_seq = diag_seq[1:]        # remaining diagnoses = input

        # Sort input by frequency (most → least)
        input_seq = sorted(
            input_seq,
            key=lambda x: diag_freq.get(idx2diag.get(x, ""), 0),
            reverse=True
        )

        # Truncate/pad input sequence
        input_seq = input_seq[:input_length]
        pad_len = input_length - len(input_seq)
        input_seq = input_seq + [0] * pad_len  # 0 is padding token

        inputs.append(input_seq)
        outputs.append(target)

        # Record corresponding SEX_CODE
        genders.append(df_subset["SEX_CODE"].iloc[idx])

    # 5. Preprocess inputs and outputs
    inputs = np.array(inputs)
    outputs = to_categorical(outputs, num_classes=len(tokenizer.word_index) + 1)

    # Predict using the trained model
    y_pred = np.argmax(model.predict(inputs), axis=1)
    y_true = np.argmax(outputs, axis=1)

    # Create predictions DataFrame
    pred_df = pd.DataFrame({
            "SEX_CODE": genders,
            "outputs_true": y_true,
            "outputs_pred": y_pred
    })

    pred_df.to_csv(output_file, index=False)
    print(f"Saved synthetic predictions to {output_file}")

###############################
# E. Main Experiment Flow
###############################
# Preprocess real data
X, y, genders, tokenizer, vocab_size = preprocess_data(real_dataset_path, real_action_column, input_length)

# === 6. Data splitting ===
# (a) --- Train on 80% real and test on 20% real ---
X_train, X_test, y_train, y_test, gender_train, gender_test = train_test_split(
    X, y, genders, test_size=0.2, random_state=42
)

# === 9. Train the models ===
# === RNN TRAINING ===
print("Training RNN...")
rnn_model = build_rnn_model(vocab_size, input_length, embedding_dim, rnn_units, dropout_rate, learning_rate)
rnn_model.summary()

# Train the model on real data
rnn_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

# Evaluate on the real test set and save predictions
rnn_real_output = os.path.join(output_path, "DIAG_wide_RNN_real_predictions.csv")
evaluate_and_save_predictions(rnn_model, X_test, y_test, gender_test, rnn_real_output)

# (b) --- Test on 100% synthetic files ---
for case in synthetic_cases:
  for seed in seeds:
    filename = f"{case['model']}_output_{case['dataset']}_{seed}.csv"
    filepath = os.path.join(case["folder"], filename)

    # Check if the input file exists
    if not os.path.exists(filepath):
      print(f"File {filepath} does not exist. Skipping.")
      continue

    # Predict for RNN
    rnn_out = os.path.join(output_path, f"{case['model']}_output_{case['dataset']}_RNN_predictions_seed_{seed}.csv")
    predict_on_synthetic(rnn_model, tokenizer, filepath, case["column"], rnn_out, input_length)
