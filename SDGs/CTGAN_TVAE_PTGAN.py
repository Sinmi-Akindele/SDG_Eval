# Variable declaration
basePath = #INSERT PATH
filePath = #INSERT PATH
seeds = [42, 50, 61, 79, 83]

# Import methods
import os
import sdv
import pandas as pd
from snsynth import Synthesizer
from sdv.metadata import Metadata
from sdv.sequential import PARSynthesizer
from sdv.single_table import TVAESynthesizer
from sdv.single_table import CTGANSynthesizer

# ________________Function declartion________________
# Generate synthetic data for CTGAN
def CTGAN_synthetic_data(df, num_samples):

    # Add the table to the metadata
    metadata = Metadata.detect_from_dataframe(data=df)

     # Create and train the CTGAN model
    ctgan = CTGANSynthesizer(metadata,
                             enforce_min_max_values=True,
                             enforce_rounding=True,
                             epochs=1000,
                             verbose=True)
    ctgan.fit(df)

    # Create synthetic data
    synthetic_data = ctgan.sample(num_samples)

    return synthetic_data

# Generate synthetic data for TVAE
def TVAE_synthetic_data(df, num_samples):

    # Create a blank SingleTableMetadata
    metadata = Metadata.detect_from_dataframe(data=df)

    # Create and train the TVAE model
    tvae = TVAESynthesizer(metadata,
                           enforce_min_max_values=True,
                           enforce_rounding=True,
                           epochs=900,
                           verbose=True)
    tvae.fit(df)

    # Create synthetic data
    synthetic_data = tvae.sample(num_samples)

    return synthetic_data

# Generate synthetic data for PATECTGAN
def PATECTGAN_synthetic_data(df, num_samples):

    # Create and train the PATECTGAN model
    patectgan = Synthesizer.create("patectgan",
                               epsilon=3.0,
                               verbose=True)

    patectgan.fit(df, preprocessor_eps=1.0)

    # Create synthetic data
    synthetic_data = patectgan.sample(num_samples)

    return synthetic_data

# ________________File generation________________
# Generate synthetic data files for CTGAN, TVAE, PATECTGAN, and PARSynthesizer
def generate_synthetic_data_files(df, model_func, num_samples, seeds, path, file_prefix):
    for seed in seeds:
        # Set the random seed for reproducibility
        if hasattr(model_func, 'set_random_state'):
            model_func.set_random_state(seed)

        # Generate synthetic data
        synthetic_data = model_func(df, num_samples)

        # Save synthetic data to CSV
        filename = f"{path}{file_prefix}_{seed}.csv"
        synthetic_data.to_csv(filename, index=False)
        print(f"Generated and saved {filename}")

# Define the models and datasets
models_and_datasets = {
    #'CTGAN': ["adult_data", "adult_test","compas-scores-two-years", "german_data"],
    #'TVAE': ["adult_data", "adult_test","compas-scores-two-years", "german_data"],
    #'PATECTGAN': ["adult_data", "adult_test","compas-scores-two-years", "german_data"]
    'CTGAN': ["adult", "australian", "GermanCredit_age25", "compas-scores", "ACSHI", "practitioner_information"],
    'TVAE': ["adult", "australian", "GermanCredit_age25", "compas-scores", "ACSHI", "practitioner_information"],
    'PATECTGAN': ["adult", "australian", "GermanCredit_age25", "compas-scores", "ACSHI", "practitioner_information"]
    #'PAR': ["PIAAC"],
    #'CTGAN': ["PIAAC_auto_encoding"]
}

# Define model functions mapping
model_functions = {
    'CTGAN': CTGAN_synthetic_data,
    'TVAE': TVAE_synthetic_data,
    'PATECTGAN': PATECTGAN_synthetic_data
    #'PAR': PAR_synthetic_data
}

# Loop over models and datasets
for model_name, datasets in models_and_datasets.items():
    model_func = model_functions[model_name]
    for dataset_name in datasets:
        # Load dataset as pandas dataframe
        df = pd.read_csv(basePath + dataset_name + ".csv")

        # Determine the number of samples
        num_samples = len(df)

        # Generate synthetic data
        generate_synthetic_data_files(df, model_func, num_samples, seeds, filePath, file_prefix=f"{model_name}_output_{dataset_name}")
        print("File created successfully!")
