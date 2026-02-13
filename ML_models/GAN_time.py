# Imports
import os
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Enable eager execution
tf.compat.v1.enable_eager_execution()

# Config
synthetic_models = ["timegan", "tsdiff"]
dataset_names = ["MetroTraffic", "Energy", "Tesla"]
trials = [0, 1, 2, 3, 4]
model_types = ["gan"]
seq_len = 24

basePath = # INSERT PATH
synthPath = # INSERT PATH
filePath = # INSERT PATH

# Helper Functions
def min_max_normalize(df):
    return (df - df.min()) / (df.max() - df.min() + 1e-7)

def make_sequences(data, seq_len):
    sequences = []
    for i in range(len(data) - seq_len + 1):
        sequences.append(data[i:i + seq_len].to_numpy())
    return np.array(sequences)

# GAN Predictive Score
def predictive_score(ori_data, generated_data, test_type="real"):
    generated_data = np.array(generated_data, dtype=np.float32)
    _, _, dim = generated_data.shape
    hidden_dim = dim // 2
    epochs = 30
    batch_size = 128

    if test_type == "real":
        ori_data = np.array(ori_data, dtype=np.float32)
        X_train = generated_data[:, :-1, :]
        Y_train = generated_data[:, -1, :]
        X_test = ori_data[:, :-1, :]
        Y_test = ori_data[:, -1, :]

    elif test_type == "synth":
        train_idx, test_idx = train_test_split(
            np.arange(len(generated_data)), test_size=0.2, random_state=42
        )
        X_train = generated_data[train_idx, :-1, :]
        Y_train = generated_data[train_idx, -1, :]
        X_test = generated_data[test_idx, :-1, :]
        Y_test = generated_data[test_idx, -1, :]

    else:
        raise ValueError("Unknown test_type")

    # Generator
    generator = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(seq_len - 1, dim)),
        tf.keras.layers.GRU(hidden_dim),
        tf.keras.layers.Dense(dim, activation="sigmoid")
    ])

    # Discriminator
    discriminator = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(seq_len, dim)),
        tf.keras.layers.GRU(hidden_dim),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    g_optimizer = tf.keras.optimizers.Adam(1e-4)
    d_optimizer = tf.keras.optimizers.Adam(1e-4)
    bce = tf.keras.losses.BinaryCrossentropy()

    @tf.function
    def train_step(x, y):
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)

        # Discriminator
        with tf.GradientTape() as d_tape:
            real_seq = tf.concat([x, tf.expand_dims(y, axis=1)], axis=1)
            fake_y = generator(x, training=True)
            fake_seq = tf.concat([x, tf.expand_dims(fake_y, axis=1)], axis=1)

            d_real = discriminator(real_seq, training=True)
            d_fake = discriminator(fake_seq, training=True)

            d_loss = (
                bce(tf.ones_like(d_real), d_real) +
                bce(tf.zeros_like(d_fake), d_fake)
            )

        d_grads = d_tape.gradient(d_loss, discriminator.trainable_variables)
        d_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))

        # Generator
        with tf.GradientTape() as g_tape:
            fake_y = generator(x, training=True)
            fake_seq = tf.concat([x, tf.expand_dims(fake_y, axis=1)], axis=1)
            d_fake = discriminator(fake_seq, training=True)

            adv_loss = bce(tf.ones_like(d_fake), d_fake)
            l1_loss = tf.reduce_mean(tf.abs(y - fake_y))
            g_loss = adv_loss + l1_loss

        g_grads = g_tape.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))

    # Train
    dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    dataset = dataset.shuffle(1024).batch(batch_size)

    for _ in range(epochs):
        for x_batch, y_batch in dataset:
            train_step(x_batch, y_batch)

    # Evaluate MAE (per sequence)
    Y_pred = generator.predict(X_test, verbose=0)

    mae_list = [
        mean_absolute_error(Y_test[i], Y_pred[i])
        for i in range(len(Y_test))
    ]

    return np.mean(mae_list)

# Loop Through All Combinations
results = []

# Train on real, test on real and synthetic
for dataset in dataset_names:
    # Load and preprocess real data
    real_path = os.path.join(basePath, f"{dataset}.csv")
    real_df = pd.read_csv(real_path)

    if dataset == "MetroTraffic":
        real_df = real_df.drop(columns=['holiday', 'weather_main', 'weather_description', 'date_time'])
    elif dataset == "Energy":
        real_df = real_df.drop(columns=['date'])
    elif dataset == "Tesla":
        real_df = real_df.drop(columns=['Id', 'Date'])

    real_df = min_max_normalize(real_df)
    real_seqs = make_sequences(real_df, seq_len)

    # Split once for consistency
    train_idx, test_idx = train_test_split(
        np.arange(len(real_seqs)), test_size=0.2, random_state=42
    )
    real_train = real_seqs[train_idx]
    real_test = real_seqs[test_idx]

    # Train on real -> test on real
    mae_real = predictive_score(real_test, real_train, test_type="real")
    pd.DataFrame([{"mae": mae_real}]).to_csv(
        os.path.join(filePath, f"{dataset}_gan_real_predictions.csv"), index=False
    )

    for synth_model in synthetic_models:
        for trial in trials:
            synth_path = os.path.join(
                synthPath, f"{synth_model}_output_{dataset}_trial_{trial}.csv"
            )
            if not os.path.exists(synth_path):
                continue

            synth_df = pd.read_csv(synth_path)
            synth_df = min_max_normalize(synth_df)
            synth_seqs = make_sequences(synth_df, seq_len)

            mae_synth = predictive_score(real_train, synth_seqs, test_type="synth")
            pd.DataFrame([{"mae": mae_synth}]).to_csv(
                os.path.join(
                    filePath,
                    f"{synth_model}_{dataset}_gan_predictions_trial_{trial}.csv"
                ),
                index=False
            )
