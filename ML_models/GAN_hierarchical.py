# Import methods
import os
import torch
import numpy as np
import pandas as pd
from torch import nn
from sklearn.metrics import accuracy_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Embedding
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Set the base path and parameters
basePath = # INSERT PATH
real_dataset_path = # INSERT PATH
real_action_column = "ALL_PROC_CODES"
output_path = os.path.join(basePath, "MlEfficacy2A")
real_df = pd.read_csv(real_dataset_path, usecols=["ALL_PROC_CODES", "ETHNICITY"])
real_df = real_df.rename(columns={"ALL_PROC_CODES": "sequence"})

synthetic_cases = [
    {"folder": os.path.join(basePath, "Synthetic"), "model": "PAR", "dataset": "PROC_wide", "column": "ALL_PROC_CODES"},
    {"folder": os.path.join(basePath, "Synthetic"), "model": "CTGAN", "dataset": "PROC_wide", "column": "ALL_PROC_CODES"},
    {"folder": os.path.join(basePath, "Synthetic"), "model": "CTGAN", "dataset": "PROC_numeric_decoding", "column": "PS_Numeric_Decoding"},
    {"folder": os.path.join(basePath, "Synthetic"), "model": "CTGAN", "dataset": "PROC_auto_decoding", "column": "PS_Auto_Decoding"}
]

seeds = [42, 50, 61, 79, 83]

embedding_dim = 128
hidden_dim = 128
noise_dim = 32
epochs = 150
batch_size = 64
max_len = 10

# Step 1: Data type conversion
def convert_delimiter(data):
    data = data.dropna()
    data["sequence"] = data["sequence"].str.replace(",", " ")
    return data

# Step 2: Tokenization
def tokenize(data):
    return data["sequence"].apply(lambda x: x.strip().split())

# Step 3: Encoding
def encode_tokens(tokenized_seqs):
    le = LabelEncoder()
    all_tokens = [token for seq in tokenized_seqs for token in seq]
    le.fit(all_tokens)
    encoded = tokenized_seqs.apply(lambda seq: le.transform(seq))
    encoded = encoded.apply(lambda x: x.tolist())
    return encoded, le

# Step 4: Sequence Preparation
def prepare_sequences(encoded_seqs, vocab_size, max_len):
    inputs, targets = [], []
    
    for proc_seq in encoded_seqs:
        if len(proc_seq) < 2:
            continue  # cannot predict if sequence too short
        
        target = proc_seq[0]  # first token = most recent diagnosis
        input_seq = proc_seq[1:]  # tokens 1 to max_len
        
        # Truncate/pad input sequence
        input_seq = input_seq[:max_len]
        pad_len = max_len - len(input_seq)
        input_seq = input_seq + [0] * pad_len  # 0 is padding token
        
        inputs.append(input_seq)
        targets.append(target)
    
    # Preprocess inputs and outputs
    inputs = np.array(inputs)
    #targets = to_categorical(targets, num_classes=vocab_size)
    targets = np.array(targets)
    
    return inputs, targets

# Step 6: Dataset class
class ActionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Step 6a: Generator
class Generator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, noise_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim + noise_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, noise):
        x = self.embedding(x)
        noise = noise.unsqueeze(1).repeat(1, x.size(1), 1)
        x = torch.cat((x, noise), dim=2)
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out  # logits

# Step 6b: Discriminator
class Discriminator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return torch.sigmoid(out)

def evaluate_and_save_predictions(model, X_test, y_test, gender_test, output_file, noise_dim, device):
    """
    Evaluates the generator on the test set and saves predictions to a CSV file.
    
    Parameters:
    - model: Trained Generator model
    - X_test (ndarray): Test input sequences (shape: [N, 20])
    - y_test (ndarray): True labels (shape: [N])
    - gender_test (list or Series): Corresponding gender info
    - output_file (str): File path to save predictions
    - noise_dim (int): Dimensionality of the noise vector
    - device: torch.device (CPU or CUDA)
    """
    model.eval()
    all_preds = []
    
    X_tensor = torch.tensor(X_test, dtype=torch.long).to(device)
    y_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
    num_samples = X_tensor.size(0)
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            X_batch = X_tensor[i:i + batch_size]
            y_batch = y_tensor[i:i + batch_size]
            gender_batch = gender_test[i:i + batch_size]
            
            noise = torch.randn(X_batch.size(0), noise_dim).to(device)
            output_logits = model(X_batch, noise)
            preds = torch.argmax(output_logits, dim=1).cpu().numpy()
            true_labels = y_batch.cpu().numpy()
            
            # Append predictions
            for g, y_t, y_p in zip(gender_batch, true_labels, preds):
                all_preds.append((g, y_t, y_p))

    # Save to CSV
    pred_df = pd.DataFrame(all_preds, columns=["ETHNICITY", "outputs_true", "outputs_pred"])
    pred_df.to_csv(output_file, index=False)
    print(f"Saved predictions to {output_file}")

def predict_on_synthetic(model, le, synth_data_path, action_column, output_file, noise_dim, device, max_len):
    """
    Processes synthetic dataset and predicts the 21st action using the trained Generator model.
    
    Parameters:
    - model: Trained Generator (PyTorch model)
    - le: Fitted LabelEncoder from training data
    - synth_data_path: Path to the synthetic CSV
    - action_column: Name of column containing synthetic action sequences
    - output_file: Path to output CSV with predictions
    - noise_dim: Dimensionality of the noise vector
    - device: CPU or CUDA device
    """
    df = pd.read_csv(synth_data_path)

    # Step 1: Convert delimiters
    df = df.rename(columns={action_column: "sequence"})
    df = convert_delimiter(df)  # replaces commas with spaces

    # Step 2: Tokenization
    tokenized = tokenize(df)

    # Step 3: Encoding using pretrained label encoder
    #encoded = tokenized.apply(lambda seq: le.transform(seq) if all(t in le.classes_ for t in seq) else [])
    encoded = tokenized.apply(lambda seq: le.transform(seq).tolist() if all(t in le.classes_ for t in seq) else [])

    # Step 4: Sequence preparation (first 20 as input, 21st as target if available)
    inputs, targets = [], []
    genders = []
    for idx, proc_seq in enumerate(encoded):
        if len(proc_seq) < 2:
            continue  # cannot predict if sequence too short
        
        target = proc_seq[0]  # first token = most recent diagnosis
        input_seq = proc_seq[1:]  # tokens 1 to max_len
        
        # Truncate/pad input sequence
        input_seq = input_seq[:max_len]
        pad_len = max_len - len(input_seq)
        input_seq = input_seq + [0] * pad_len  # 0 is padding token
        
        inputs.append(input_seq)
        targets.append(target)
        # Record corresponding ETHNICITY
        genders.append(df["ETHNICITY"].iloc[idx])

    if len(inputs) == 0:
        print("No valid sequences found in synthetic data.")
        return

    # Convert to tensors
    inputs_tensor = torch.tensor(inputs, dtype=torch.long).to(device)
    targets_tensor = torch.tensor(targets, dtype=torch.long).to(device)

    model.eval()
    all_preds = []

    with torch.no_grad():
        for i in range(0, len(inputs), batch_size):
            X_batch = inputs_tensor[i:i + batch_size]
            y_batch = targets_tensor[i:i + batch_size]
            g_batch = genders[i:i + batch_size]

            noise = torch.randn(X_batch.size(0), noise_dim).to(device)
            logits = model(X_batch, noise)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_true = y_batch.cpu().numpy()

            for g, yt, yp in zip(g_batch, y_true, preds):
                all_preds.append((g, yt, yp))

    # Save predictions
    pred_df = pd.DataFrame(all_preds, columns=["ETHNICITY", "outputs_true", "outputs_pred"])
    pred_df.to_csv(output_file, index=False)
    print(f"Saved synthetic predictions to {output_file}")

# E. Main Experiment Flow (GAN)
# === A. Preprocess Real Data ===
print("Preprocessing real data...")
real_df = convert_delimiter(real_df)
tokenized_seqs = tokenize(real_df)
encoded_seqs, le = encode_tokens(tokenized_seqs)

valid_indices = [i for i, proc_seq in enumerate(encoded_seqs) if len(proc_seq) >= 2]
encoded_seqs = [encoded_seqs[i] for i in valid_indices]
gender_labels = real_df["ETHNICITY"].iloc[valid_indices].reset_index(drop=True)

vocab_size = len(le.classes_)
X, y = prepare_sequences(encoded_seqs, vocab_size, max_len)

# Data splitting
X_train, X_test, y_train, y_test, gender_train, gender_test = train_test_split(
    X, y, gender_labels, test_size=0.2, random_state=42
)

# === B. Build and Train GAN Models ===
print("Initializing GAN...")
train_dataset = ActionDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = ActionDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

G = Generator(vocab_size, embedding_dim, hidden_dim, noise_dim).to(device)
D = Discriminator(vocab_size, embedding_dim, hidden_dim).to(device)

criterion_adv = nn.BCELoss()
criterion_cls = nn.CrossEntropyLoss()
optimizer_G = torch.optim.Adam(G.parameters(), lr=1e-4)
optimizer_D = torch.optim.Adam(D.parameters(), lr=1e-4)

# === Training Loop ===
print("Training GAN...")
for epoch in range(epochs):
    G.train()
    D.train()

    for real_seqs, real_actions in train_loader:
        real_seqs = real_seqs.to(device)
        real_actions = real_actions.to(device)

        batch_size = real_seqs.size(0)
        noise = torch.randn(batch_size, noise_dim).to(device)

        # === Discriminator ===
        real_labels = torch.ones(batch_size, 1).to(device) * 0.9
        fake_labels = torch.zeros(batch_size, 1).to(device)

        D_real_input = torch.cat((real_seqs, real_actions.unsqueeze(1)), dim=1)
        D_real_output = D(D_real_input)
        loss_real = criterion_adv(D_real_output, real_labels)

        fake_logits = G(real_seqs, noise)
        fake_actions = torch.argmax(fake_logits, dim=1)
        D_fake_input = torch.cat((real_seqs, fake_actions.unsqueeze(1)), dim=1)
        D_fake_output = D(D_fake_input.detach())
        loss_fake = criterion_adv(D_fake_output, fake_labels)

        loss_D = loss_real + loss_fake
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # === Generator ===
        fake_logits = G(real_seqs, noise)
        fake_actions = torch.argmax(fake_logits, dim=1)
        D_fake_input = torch.cat((real_seqs, fake_actions.unsqueeze(1)), dim=1)
        D_output = D(D_fake_input)

        loss_G_adv = criterion_adv(D_output, real_labels)
        loss_G_cls = criterion_cls(fake_logits, real_actions)
        loss_G = loss_G_adv + loss_G_cls

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

    print(f"Epoch {epoch+1}/{epochs} | D Loss: {loss_D.item():.4f} | G Loss: {loss_G.item():.4f}")

# === C. Evaluate Test Accuracy ===
print("Evaluating Test Accuracy...")
G.eval()
correct = 0
total = 0

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        noise = torch.randn(X_batch.size(0), noise_dim).to(device)
        output = G(X_batch, noise)
        preds = torch.argmax(output, dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")

# === D. Save Real Data Predictions ===
gan_real_output = os.path.join(output_path, "PROC_wide_GAN_real_predictions.csv")
evaluate_and_save_predictions(G, X_test, y_test, gender_test, gan_real_output, noise_dim, device)

# === E. Evaluate on 100% Synthetic Datasets ===
for case in synthetic_cases:
    for seed in seeds:
        filename = f"{case['model']}_output_{case['dataset']}_{seed}.csv"
        filepath = os.path.join(case["folder"], filename)

        if not os.path.exists(filepath):
            print(f"File {filepath} does not exist. Skipping.")
            continue

        gan_out = os.path.join(output_path, f"{case['model']}_output_{case['dataset']}_GAN_predictions_seed_{seed}.csv")
        predict_on_synthetic(
            model=G,
            le=le,
            synth_data_path=filepath,
            action_column=case["column"],
            output_file=gan_out,
            noise_dim=noise_dim,
            device=device,
	    max_len=max_len
        )
