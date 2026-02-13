# Import methods
import os
import sdv
import pandas as pd
from snsynth import Synthesizer
from sdv.metadata import Metadata
from sdv.sequential import PARSynthesizer
from sdv.single_table import CTGANSynthesizer

# Variable declaration
basePath = # INSERT PATH
filePath = # INSERT PATH
seeds = [42, 50, 61, 79, 83]

# ________________Function declartion________________
# Generate synthetic data for CTGAN
def CTGAN_synthetic_data(df, num_samples, dataset_name):
    # Determine which column to treat as categorical based on dataset_name
    if dataset_name == "PROC_auto_encoding":
        sequence_column = "PS_Auto_Encoding"
    elif dataset_name == "PROC_numeric_encoding":
        sequence_column = "PS_Numeric_Encoding"
    elif dataset_name == "PROC_wide":
        sequence_column = "ALL_PROC_CODES"
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    # Detect metadata from the dataframe
    metadata = Metadata.detect_from_dataframe(data=df)

    # Explicitly mark the target sequence column as categorical
    metadata.update_column(
        column_name=sequence_column,
        sdtype='categorical'
    )

    # Create and train the CTGAN model
    ctgan = CTGANSynthesizer(
        metadata,
        enforce_min_max_values=True,
        enforce_rounding=True,
        epochs=1000,
        verbose=True
    )
    ctgan.fit(df)

    # Create synthetic data
    synthetic_data = ctgan.sample(num_samples)

    return synthetic_data

# Generate synthetic data for PAR
def PAR_synthetic_data(df, num_samples, dataset_name):
    # Detect metadata from the input DataFrame
    metadata = Metadata.detect_from_dataframe(data=df)

    # Update metadata for each column
    metadata.update_column(
        column_name="ID",
        sdtype="id")
    metadata.update_column(
        column_name="SEX_CODE",
        sdtype="categorical")
    metadata.update_column(
        column_name="PAT_AGE",
        sdtype="categorical")
    metadata.update_column(
        column_name="RACE",
        sdtype="categorical")
    metadata.update_column(
        column_name="ETHNICITY",
        sdtype="categorical")
    metadata.update_column(
        column_name="PAT_ZIP",
        sdtype="numerical",
        computer_representation="Int64")
    metadata.update_column(
        column_name="ALL_PROC_CODES",
        sdtype="categorical")
    metadata.update_column(
        column_name="FIRST_PAYMENT_SRC",
        sdtype="categorical")
    metadata.update_column(
        column_name="PHARM_AMOUNT",
        sdtype="numerical",
        computer_representation="Float")
    metadata.update_column(
        column_name="MEDSURG_AMOUNT",
        sdtype="numerical",
        computer_representation="Float")
    metadata.update_column(
        column_name="OP_AMOUNT",
        sdtype="numerical",
        computer_representation="Int64")
    metadata.update_column(
        column_name="TOTAL_CHARGES",
        sdtype="numerical",
        computer_representation="Float")
    metadata.update_column(
        column_name="ActionSeqOrder",
        sdtype="numerical",
        computer_representation="Int64")

    # Set sequence key and index
    metadata.set_sequence_key(column_name='ID')
    metadata.set_sequence_index(column_name='ActionSeqOrder')

    # Remove primary key from metadata if it exists
    metadata.remove_primary_key()

    # Initialize synthesizer with metadata
    synthesizer = PARSynthesizer(
        metadata,
        context_columns=['SEX_CODE', 'PAT_AGE', 'RACE', 'ETHNICITY',    'PAT_ZIP',
                         'FIRST_PAYMENT_SRC',   'PHARM_AMOUNT', 'MEDSURG_AMOUNT',
                         'OP_AMOUNT',   'TOTAL_CHARGES'],
        verbose=True)

    # Fit synthesizer to the input data
    synthesizer.fit(df)

    # Generate synthetic data
    synthetic_data = synthesizer.sample(num_samples)

    return synthetic_data

# ________________File generation________________
# Generate synthetic data files for CTGAN, TVAE, PATECTGAN, and PARSynthesizer
def generate_synthetic_data_files(df, model_func, num_samples, seeds, path, file_prefix, dataset_name):
    for seed in seeds:
        # Set the random seed for reproducibility
        if hasattr(model_func, 'set_random_state'):
            model_func.set_random_state(seed)

        # Generate synthetic data
        synthetic_data = model_func(df, num_samples, dataset_name)

        # Save synthetic data to CSV
        filename = f"{path}{file_prefix}_{seed}.csv"
        synthetic_data.to_csv(filename, index=False)
        print(f"Generated and saved {filename}")

# Define the models and datasets
models_and_datasets = {
    'PAR': ["PROC_long"],
    'CTGAN': ["PROC_wide", "PROC_numeric_encoding", "PROC_auto_encoding"]
}

# Define the model functions mapping
model_functions = {
    'CTGAN': CTGAN_synthetic_data,
    'PAR': PAR_synthetic_data
}

# Loop over models and datasets
for model_name, datasets in models_and_datasets.items():
    model_func = model_functions[model_name]
    for dataset_name in datasets:
        # Load dataset
        df = pd.read_csv(basePath + dataset_name + ".csv")

        # Determine number of samples
        num_samples = len(df)

        # Generate synthetic data
        generate_synthetic_data_files(df, model_func, num_samples, seeds, filePath, f"{model_name}_output_{dataset_name}", dataset_name)
        print("File created successfully!")
