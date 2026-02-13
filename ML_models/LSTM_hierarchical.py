# Import methods
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Set the base path and parameters
basePath = # INSERT PATH
real_dataset_path = # INSERT PATH
real_action_column = "ALL_PROC_CODES"
output_path = os.path.join(basePath, "MlEfficacy2A")

synthetic_cases = [
    {"folder": os.path.join(basePath, "Synthetic"), "model": "PAR", "dataset": "PROC_wide", "column": "ALL_PROC_CODES"},
    {"folder": os.path.join(basePath, "Synthetic"), "model": "CTGAN", "dataset": "PROC_wide", "column": "ALL_PROC_CODES"},
    {"folder": os.path.join(basePath, "Synthetic"), "model": "CTGAN", "dataset": "PROC_numeric_decoding", "column": "PS_Numeric_Decoding"},
    {"folder": os.path.join(basePath, "Synthetic"), "model": "CTGAN", "dataset": "PROC_auto_decoding", "column": "PS_Auto_Decoding"}
]

# Variable declaration
seeds = [42, 50, 61, 79, 83]

# Hyper-parameters
input_length=20
embedding_dim=60
lstm_units=64
dropout_rate=0.1
learning_rate=0.001
batch_size=32
epochs=50

# A. Data Preprocessing Function
def preprocess_data(data, action_column):
    """
    Loads the dataset, converts the given action sequence column from comma-separated to space-separated,
    tokenizes the action sequences, and creates input-output pairs (first 4 tokens as input, 5th as output).

    Returns:
      - inputs: numpy array of input sequences (each of length 4)
      - outputs: one-hot encoded outputs (5th token)
      - genders: list of GENDER_R values (if keep_gender is True; otherwise an empty list)
      - tokenizer: fitted Keras Tokenizer
      - vocab_size: size of the vocabulary
    """
    # === 1. Data type conversion ===
    # Load dataset
    df = pd.read_csv(data)
    # Convert the action sequence from comma-separated to space-separated
    df[action_column] = df[action_column].apply(lambda x: " ".join(x.split(",")))

    # Keep only the columns we need - gender and action_column
    df = df[["ETHNICITY", action_column]]

    # === 2. Tokenization ===
    # Use the Keras Tokenizer to build a vocabulary of unique actions.
    tokenizer = Tokenizer(lower=False, filters='')
    tokenizer.fit_on_texts(df[action_column])
    vocab_size = len(tokenizer.word_index) + 1  # +1 for padding token

    # === 3. Encode sequences ===
    # Convert each formatted sequence into a list of token IDs
    df["TokenizedSequence"] = tokenizer.texts_to_sequences(df[action_column])

    # === 4. Sequence preparation: create input-output pairs ===
    # For each sequence, the input is the first four tokens
    # The output is the fifth token.
    inputs, outputs, genders = [], [], []

    # For each sequence, we use the first twenty tokens as input and the twenty-first token as output.
    for idx, seq in enumerate(df["TokenizedSequence"]):
        if len(seq) < 5:
            continue  # Skip sequences that are too short to form an input-output pair
        inputs.append(seq[:4])
        outputs.append(seq[4])
        # Record corresponding GENDER_R
        genders.append(df["ETHNICITY"].iloc[idx])

    # === 5. Preprocess inputs and outputs ===
    # Convert inputs to numpy array
    inputs = np.array(inputs)

    # Convert outputs to one-hot encoding since model will use categorical_cross entropy.
    outputs = to_categorical(outputs, num_classes=vocab_size)

    return inputs, outputs, genders, tokenizer, vocab_size
  
# B. Model Building Function
def build_lstm_model(vocab_size, input_length, embedding_dim, lstm_units, dropout_rate, learning_rate):
    # Build the LSTM model
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length),
        LSTM(lstm_units),
        Dropout(dropout_rate),
        Dense(vocab_size, activation='softmax')
    ])

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate), metrics=['accuracy'])
    return model

# C. Evaluation & Prediction Saving Function
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
            "ETHNICITY": gender_test,
            "outputs_true": y_true,
            "outputs_pred": y_pred
    })

    # Save the predictions to a CSV file
    pred_df.to_csv(output_file, index=False)
    print(f"Saved predictions to {output_file}")

# D. Function to Run Predictions on Synthetic Files
def predict_on_synthetic(model, tokenizer, synth_data, action_column, output_file):
    """
    Loads a synthetic dataset, processes the action sequence column, creates input sequences (first 4 tokens),
    uses the trained model to predict the next action, and saves the predictions.

    The output CSV will have:
      - GENDER_R (if available)
      - outputs_true (if applicable, i.e. the 5th token if present)
      - outputs_pred (predicted next token)
    """
    df = pd.read_csv(synth_data)

    # Convert the action sequence from comma-separated to space-separated
    df[action_column] = df[action_column].apply(lambda x: " ".join(x.split(",")))

    # Keep only the columns we need - gender and action_column
    df_subset = df[["ETHNICITY", action_column]]

    # Tokenize using the existing tokenizer
    df_subset["TokenizedSequence"] = tokenizer.texts_to_sequences(df_subset[action_column])

    inputs, outputs, genders = [], [], []
    # Create input-output pairs from the synthetic data, if possible.
    for idx, seq in enumerate(df_subset["TokenizedSequence"]):
        if len(seq) < 5:
            continue
        inputs.append(seq[:4])
        outputs.append(seq[4])
        genders.append(df_subset["ETHNICITY"].iloc[idx])

    inputs = np.array(inputs)
    outputs = to_categorical(outputs, num_classes=len(tokenizer.word_index) + 1)

    # Predict using the trained model
    y_pred = np.argmax(model.predict(inputs), axis=1)
    y_true = np.argmax(outputs, axis=1)

    # Create predictions DataFrame
    pred_df = pd.DataFrame({
            "ETHNICITY": genders,
            "outputs_true": y_true,
            "outputs_pred": y_pred
    })

    pred_df.to_csv(output_file, index=False)
    print(f"Saved synthetic predictions to {output_file}")
  
# E. Main Experiment Flow
# Preprocess real data
X, y, genders, tokenizer, vocab_size = preprocess_data(real_dataset_path, real_action_column)

# === 6. Data splitting ===
# (a) --- Train on 80% real and test on 20% real ---
X_train, X_test, y_train, y_test, gender_train, gender_test = train_test_split(
    X, y, genders, test_size=0.2, random_state=42
)

# 9. Train the models
# LSTM TRAINING
print("Training LSTM...")
lstm_model = build_lstm_model(vocab_size, input_length, embedding_dim, lstm_units, dropout_rate, learning_rate)
lstm_model.summary()

# Train the model on real data
lstm_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

# Evaluate on the real test set and save predictions
lstm_real_output = os.path.join(output_path, "PROC_wide_LSTM_real_predictions.csv")
evaluate_and_save_predictions(lstm_model, X_test, y_test, gender_test, lstm_real_output)

# (b) --- Test on 100% synthetic files ---
for case in synthetic_cases:
  for seed in seeds:
    filename = f"{case['model']}_output_{case['dataset']}_{seed}.csv"
    filepath = os.path.join(case["folder"], filename)

    # Check if the input file exists
    if not os.path.exists(filepath):
      print(f"File {filepath} does not exist. Skipping.")
      continue

    # Predict for LSTM
    lstm_out = os.path.join(output_path, f"{case['model']}_output_{case['dataset']}_LSTM_predictions_seed_{seed}.csv")
    predict_on_synthetic(lstm_model, tokenizer, filepath, case["column"], lstm_out)
