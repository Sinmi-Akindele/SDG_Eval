# Import methods
import os
import time
import torch
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import torch.utils.data
import torch.optim as optim
from torch.optim import Adam
warnings.filterwarnings("ignore")
from sklearn import preprocessing
from sklearn import model_selection
from torch.nn import functional as F
from typing import List, Tuple, Optional, Union
from sklearn.mixture import BayesianGaussianMixture
from torch.nn import (
    Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, Conv2d, Conv1d,
    ConvTranspose2d, ConvTranspose1d, Sigmoid, init, BCELoss, CrossEntropyLoss,
    SmoothL1Loss, LayerNorm
)


# ________________Variable declartion________________
basePath = # INSERT PATH
filePath = # INSERT PATH
seeds = [42, 50, 61, 79, 83]
datasets = [
    "adult", "german_data", "compas-scores-two-years"
]
dataset_configs = {
    "adult": {
        "categorical_columns": [
            'workclass', 'education', 'marital-status', 'occupation',
            'relationship', 'race', 'sex', 'native-country', 'income-per-year'
        ],
        "log_columns": [],
        "mixed_columns": {'capital-loss':[0.0],'capital-gain':[0.0]},
        "general_columns": ["age"],
        "non_categorical_columns": [],
        "integer_columns": [
            'age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'
        ],
        "problem_type": {"Classification": "income-per-year"}
    },

    "german_data": {
        "categorical_columns": ["status", "credit_history", "purpose", "savings", "employment", 
  				"other_debtors", "property", "installment_plans", "housing",
				"skill_level", "telephone", "foreign_worker", "credit", "sex"],
        "log_columns": ["credit_amount", "month", "age"],
        "mixed_columns": {},
        "general_columns": ["investment_as_income_percentage", "residence_since", 
			    "number_of_credits", "people_liable_for"],
        "non_categorical_columns": [],
        "integer_columns": ["month", "credit_amount", "investment_as_income_percentage",
			    "residence_since", "age", "number_of_credits", "people_liable_for"],
        "problem_type": {"Classification": "credit"}
    },

    "compas-scores-two-years": {
        "categorical_columns": [
        	"sex", "age_cat", "race", "c_charge_degree", "c_charge_desc", "two_year_recid"
        ],
        "log_columns": ["age", "juv_fel_count", "juv_misd_count", "juv_other_count", "priors_count"],
        "mixed_columns": {},
        "general_columns": [],
        "non_categorical_columns": [],
        "integer_columns": ["age", "juv_fel_count", "juv_misd_count", "juv_other_count", "priors_count"],
        "problem_type": {"Classification": "two_year_recid"}
    }
}


# ________________Function declartion________________
# Class to prepare the data based on its column type and data distribution
class DataPrep(object):
    def __init__(self, raw_df: pd.DataFrame, categorical: list, log:list, mixed:dict,
                 general:list, non_categorical:list, integer:list, type:dict, test_ratio:float):

        # Store column type information passed as arguments
        self.categorical_columns = categorical     		# purely categorical variables
        self.log_columns = log                     		# numerical variables that will be log-transformed
        self.mixed_columns = mixed                 		# mixed-type variables (categorical + numeric)
        self.general_columns = general             		# general-purpose columns (not categorical/log/mixed)
        self.non_categorical_columns = non_categorical  # variables that should be treated as numeric
        self.integer_columns = integer             		# variables that should always be integers

        # Structure to keep track of processed column types
        self.column_types = dict()
        self.column_types["categorical"] = []      # store indices of categorical columns
        self.column_types["mixed"] = {}            # store indices + placeholder values for mixed columns
        self.column_types["general"] = []          # store indices of general columns
        self.column_types["non_categorical"] = []  # store indices of numeric columns
        self.lower_bounds = {}                     # store min values for log-transformed columns
        self.label_encoder_list = []               # keep track of label encoders for categorical columns

        # Train/Test Split
        # Get the name of the target column from dict "type"
        target_col = list(type.values())[0]
        # Target variable
        y_real = raw_df[target_col]
        # Features (all other columns)
        X_real = raw_df.drop(columns=[target_col])
        # Split into train and test
        X_train_real, _, y_train_real, _ = model_selection.train_test_split(
            X_real ,y_real, test_size=test_ratio, stratify=y_real,random_state=42
        )

        # Reattach target column to the training set
        X_train_real[target_col]= y_train_real
        self.df = X_train_real

        # Handle Missing Values
        # Replace blank spaces with NaN
        self.df = self.df.replace(r' ', np.nan)
        # fill all NaNs with the string "empty"
        self.df = self.df.fillna('empty')

        # Identify columns with "empty" values
        all_columns= set(self.df.columns)
        # Categorical columns don't need this handling
        irrelevant_missing_columns = set(self.categorical_columns)
        # Only non-categorical cols do
        relevant_missing_columns = list(all_columns - irrelevant_missing_columns)

        # Replace "empty" with placeholder value (-9999999) in numeric/log/mixed columns
        for i in relevant_missing_columns:
            if i in self.log_columns:
                if "empty" in list(self.df[i].values):
                    self.df[i] = self.df[i].apply(lambda x: -9999999 if x=="empty" else x)
                    self.mixed_columns[i] = [-9999999]
            elif i in list(self.mixed_columns.keys()):
                if "empty" in list(self.df[i].values):
                    self.df[i] = self.df[i].apply(lambda x: -9999999 if x=="empty" else x )
                    self.mixed_columns[i].append(-9999999)
            else:
                if "empty" in list(self.df[i].values):
                    self.df[i] = self.df[i].apply(lambda x: -9999999 if x=="empty" else x)
                    self.mixed_columns[i] = [-9999999]

        # Log-transform log columns
        if self.log_columns:
            for log_column in self.log_columns:
                valid_indices = []
                # Collect indices of non-placeholder values
                for idx, val in enumerate(self.df[log_column].values):
                    if val!=-9999999:
                        valid_indices.append(idx)

                eps = 1
                # Compute min of valid values
                lower = np.min(self.df[log_column].iloc[valid_indices].values)
                # Store min for later inverse transformation
                self.lower_bounds[log_column] = lower

                # Different log-transform strategies depending on min value
                if lower > 0:
                    # Standard log for strictly positive values
                    self.df[log_column] = self.df[log_column].apply(
                        lambda x: np.log(x) if x != -9999999 else -9999999
                    )
                elif lower == 0:
                    # If column had zeros, add eps to avoid log(0)
                    self.df[log_column] = self.df[log_column].apply(
                        lambda x: np.log(x + eps) if x != -9999999 else -9999999
                    )
                else:
                    # If column had negative values, shift data before log
                    self.df[log_column] = self.df[log_column].apply(
                        lambda x: np.log(x - lower + eps) if x != -9999999 else -9999999
                    )

        # Encode Categorical Columns
        for column_index, column in enumerate(self.df.columns):
            if column in self.categorical_columns:
                # New LabelEncoder
                label_encoder = preprocessing.LabelEncoder()
                # Ensure values are strings
                self.df[column] = self.df[column].astype(str)
                # Fit encoder on values
                label_encoder.fit(self.df[column])

                # Save encoder for later inverse transformation
                current_label_encoder = dict()
                current_label_encoder['column'] = column
                current_label_encoder['label_encoder'] = label_encoder

                # Transform categorical column to numeric codes
                transformed_column = label_encoder.transform(self.df[column])
                self.df[column] = transformed_column

                # Store encoder + update column types
                self.label_encoder_list.append(current_label_encoder)
                self.column_types["categorical"].append(column_index)

                if column in self.general_columns:
                    self.column_types["general"].append(column_index)

                if column in self.non_categorical_columns:
                    self.column_types["non_categorical"].append(column_index)

            # If column is mixed-type, record placeholder values
            elif column in self.mixed_columns:
                self.column_types["mixed"][column_index] = self.mixed_columns[column]

            # If column is general numeric, mark it
            elif column in self.general_columns:
                self.column_types["general"].append(column_index)

        # Call parent class constructo
        super().__init__()

    def inverse_prep(self, data, eps=1):
      # Convert the input data back into a DataFrame with the same column structure as the original dataset
      df_sample = pd.DataFrame(data, columns=self.df.columns)

      # Step 1: Inverse transform categorical columns
      for i in range(len(self.label_encoder_list)):
            # Get the LabelEncoder
            le = self.label_encoder_list[i]["label_encoder"]

            # Get the column name
            col = self.label_encoder_list[i]["column"]

            # Ensure the column values are integers (LabelEncoder expects ints for inverse_transform)
            df_sample[col] = df_sample[col].astype(int)

            # Convert encoded integers back to their original categorical labels
            df_sample[col] = le.inverse_transform(df_sample[col])

      # Step 2: Inverse transform log-transformed columns
      if self.log_columns:
            # Loop over all columns in df_sample
            for i in df_sample:
                # Only process columns that were log-transformed
                if i in self.log_columns:
                    # Get the stored lower bound of the column
                    lower_bound = self.lower_bounds[i]

                    # If data was strictly positive before log, take the exponential to undo log-transform
                    if lower_bound > 0:
                        df_sample[i].apply(lambda x: np.exp(x))

                    # If data included zeros originally
                    # Shift back by subtracting eps (small correction factor)
                    # Use ceil to avoid negative numbers after transformation
                    elif lower_bound == 0:
                        df_sample[i] = df_sample[i].apply(
                            lambda x: np.ceil(np.exp(x)-eps)
                            if (np.exp(x)-eps) < 0 else (np.exp(x)-eps)
                        )

                    else:
                        # If data had negative values before log-transform,
                        # Shift back by adding the lower bound and subtracting eps
                        df_sample[i] = df_sample[i].apply(lambda x: np.exp(x)-eps+lower_bound)

      # Step 3: Ensure integer columns remain integers
      if self.integer_columns:
            for column in self.integer_columns:
                # Round values to nearest integer
                df_sample[column]= (np.round(df_sample[column].values))
                # Cast back to int type
                df_sample[column] = df_sample[column].astype(int)

      # Step 4: Replace placeholder values with NaN
      # Replace placeholder value (-9999999) with NaN (missing value)
      df_sample.replace(-999999, np.nan, inplace=True)

      # Replace any leftover "empty" strings with NaN
      df_sample.replace('empty', np.nan, inplace=True)

      # Return the fully restored DataFrame
      return df_sample


# ___________________________________________________
# Class to transform data to fit GAN input parameters
class DataTransformer():

    def __init__(self, train_data=pd.DataFrame, categorical_list=[], mixed_dict={},
                 general_list=[], non_categorical_list=[], n_clusters=10, eps=0.005): #change n_clusters back to 10

        # Placeholder for metadata
        self.meta = None

        # Number of clusters (used later for clustering or grouping logic)
        self.n_clusters = n_clusters

        # Small constant (epsilon), often used to avoid divide-by-zero errors
        # or for numerical stability in algorithms
        self.eps = eps

        # Store the training dataset (as a pandas DataFrame)
        self.train_data = train_data

        # Store list of categorical columns
        self.categorical_columns = categorical_list

        # Store dictionary of mixed columns (columns that may contain both numeric + categorical-like data)
        self.mixed_columns = mixed_dict

        # Store list of general-purpose columns (neither strictly categorical nor mixed)
        self.general_columns = general_list

        # Store list of explicitly non-categorical (numeric) columns
        self.non_categorical_columns = non_categorical_list

    def get_metadata(self):

        # Initialize an empty list to hold metadata for each column
        meta = []

        # Loop through all columns in the training dataset by index
        for index in range(self.train_data.shape[1]):
            # Extract the current column as a pandas Series
            column = self.train_data.iloc[:,index]

            # Case 1: Categorical columns
            if index in self.categorical_columns:

                # Sub-case: Categorical but also marked as non-categorical (treated as numeric instead)
                if index in self.non_categorical_columns:
                    meta.append({
                      "name": index,              # column index
                      "type": "continuous",       # treat as continuous variable
                      "min": column.min(),        # min value of the column
                      "max": column.max(),        # max value of the column
                    })

                # Sub-case: Pure categorical column
                else:
                    # list of unique categories (sorted by frequency)
                    mapper = column.value_counts().index.tolist()
                    meta.append({
                        "name": index,            # column index
                        "type": "categorical",    # mark as categorical
                        "size": len(mapper),      # number of unique categories
                        "i2s": mapper             # index-to-string mapping (for decoding categories later)
                    })

            # Case 2: Mixed columns (numeric + categorical-like placeholder values)
            elif index in self.mixed_columns.keys():
                meta.append({
                    "name": index,                      # column index
                    "type": "mixed",                    # mark as mixed type
                    "min": column.min(),                # min numeric value
                    "max": column.max(),                # max numeric value
                    "modal": self.mixed_columns[index]  # list of placeholder values (e.g., -9999999)
                })

            # Case 3: Default continuous column
            else:
                meta.append({
                    "name": index,                # column index
                    "type": "continuous",         # numeric column
                    "min": column.min(),          # min value
                    "max": column.max(),          # max value
                })

        # Return the full metadata list (one dictionary per column)
        return meta

    def fit(self):
        # Extract the raw numpy array of training data
        data = self.train_data.values

        # Build metadata for each column (from get_metadata)
        self.meta = self.get_metadata()

        # Initialize containers for fitted models and metadata
        model = []           # stores per-column models (Gaussian mixtures or None)
        self.ordering = []   # (not used here, but likely for column ordering in generation)
        self.output_info = [] # stores output structure info per column (dimensionality + activation)
        self.output_dim = 0   # total output dimension (used later for neural nets)
        self.components = []  # stores which Gaussian mixture components are active
        self.filter_arr = []  # for mixed columns, track which rows are valid numeric vs placeholder

        # Loop through each column in the metadata
        for id_, info in enumerate(self.meta):

            # Case 1: Continuous column
            if info['type'] == "continuous":

                # If it's not a "general" column
                if id_ not in self.general_columns:

                  # Fit a Bayesian Gaussian Mixture (BGM) to the column
                  gm = BayesianGaussianMixture(
                      #self.n_clusters,
                      n_components=self.n_clusters,
                      weight_concentration_prior_type='dirichlet_process',
                      weight_concentration_prior=0.001,
                      max_iter=100,
                      n_init=1,
                      random_state=42
                  )
                  gm.fit(data[:, id_].reshape([-1, 1]))

                  # Find most frequent mixture components
                  mode_freq = (pd.Series(gm.predict(data[:, id_].reshape([-1, 1]))).value_counts().keys())

                  # Save fitted model
                  model.append(gm)

                  # Identify active mixture components (weights > eps threshold)
                  old_comp = gm.weights_ > self.eps
                  comp = []
                  for i in range(self.n_clusters):
                      if (i in (mode_freq)) & old_comp[i]:
                          comp.append(True)
                      else:
                          comp.append(False)
                  self.components.append(comp)

                  # Add output info for this column:
                  # - First part: 1 value, passed through tanh (scaled continuous variable)
                  # - Second part: softmax over active components
                  self.output_info += [(1, 'tanh','no_g'), (np.sum(comp), 'softmax')]
                  self.output_dim += 1 + np.sum(comp) # increase total output dim

                else:
                  # If column is "general", skip mixture model
                  model.append(None)
                  self.components.append(None)

                  # Only include a single tanh output
                  self.output_info += [(1, 'tanh','yes_g')]
                  self.output_dim += 1

            # Case 2: Mixed column (continuous + placeholder values)
            elif info['type'] == "mixed":

                # Two mixture models:
                # gm1 = fit to all values (including placeholder/modal values)
                # gm2 = fit only to valid continuous values
                gm1 = BayesianGaussianMixture(
                    #self.n_clusters,
                    n_components=self.n_clusters,
                    weight_concentration_prior_type='dirichlet_process',
                    weight_concentration_prior=0.001,
                    max_iter=100,
                    n_init=1,
                    random_state=42)
                gm2 = BayesianGaussianMixture(
                    #self.n_clusters,
                    n_components=self.n_clusters,
                    weight_concentration_prior_type='dirichlet_process',
                    weight_concentration_prior=0.001,
                    max_iter=100,
                    n_init=1,
                    random_state=42)

                # Fit gm1 on all data in column
                gm1.fit(data[:, id_].reshape([-1, 1]))

                # Build a filter: True if value is valid (not in placeholder modal list), False otherwise
                filter_arr = []
                for element in data[:, id_]:
                    if element not in info['modal']:
                        filter_arr.append(True)
                    else:
                        filter_arr.append(False)

                # Fit gm2 on only the valid values
                gm2.fit(data[:, id_][filter_arr].reshape([-1, 1]))

                # Identify most frequent mixture components in gm2
                mode_freq = (pd.Series(gm2.predict(data[:, id_][filter_arr].reshape([-1, 1]))).value_counts().keys())

                # Store filter array for this column
                self.filter_arr.append(filter_arr)

                # Save both models for this column
                model.append((gm1,gm2))

                # Identify active components from gm2 (weights > eps and frequent enough)
                old_comp = gm2.weights_ > self.eps
                comp = []
                for i in range(self.n_clusters):
                    if (i in (mode_freq)) & old_comp[i]:
                        comp.append(True)
                    else:
                        comp.append(False)

                self.components.append(comp)

                # Output info:
                # - One tanh value for the continuous part
                # - A softmax for both mixture components and placeholder modal values
                self.output_info += [(1, 'tanh',"no_g"), (np.sum(comp) + len(info['modal']), 'softmax')]
                self.output_dim += 1 + np.sum(comp) + len(info['modal'])

            # Case 3: Categorical column
            else:
                # No mixture model, just None
                model.append(None)
                self.components.append(None)

                # Output is a softmax over all categories
                self.output_info += [(info['size'], 'softmax')]
                self.output_dim += info['size']

        # Save fitted models for all columns
        self.model = model

    def transform(self, data, ispositive = False, positive_list = None):
        values = []        # will collect transformed column encodings
        mixed_counter = 0  # keep track of how many mixed columns we’ve processed

        # Iterate through each column in metadata
        for id_, info in enumerate(self.meta):
            current = data[:, id_] # take the current column values

            # Case 1: Continuous column
            if info['type'] == "continuous":
                # If this column is not one of the "general" ones
                if id_ not in self.general_columns:
                  # Reshape into column vector
                  current = current.reshape([-1, 1])

                  # Extract GMM (Gausian Mixture Model) parameters for this column
                  means = self.model[id_].means_.reshape((1, self.n_clusters))
                  stds = np.sqrt(self.model[id_].covariances_).reshape((1, self.n_clusters))

                  # Create placeholder for feature encodings
                  features = np.empty(shape=(len(current),self.n_clusters))

                  # If we want to enforce positive encodings for some columns
                  if ispositive == True:
                      if id_ in positive_list:
                          features = np.abs(current - means) / (4 * stds)
                  else:
                    # Otherwise, normalize by subtracting mean and dividing by scaled std
                      features = (current - means) / (4 * stds)

                  # Get mixture component probabilities for each row
                  probs = self.model[id_].predict_proba(current.reshape([-1, 1]))

                  # Keep only active mixture components
                  n_opts = sum(self.components[id_])
                  features = features[:, self.components[id_]]
                  probs = probs[:, self.components[id_]]

                  # Randomly select one component for each row, according to probs
                  opt_sel = np.zeros(len(data), dtype='int')
                  for i in range(len(data)):
                      pp = probs[i] + 1e-6 # avoid division by zero
                      pp = pp / sum(pp)
                      opt_sel[i] = np.random.choice(np.arange(n_opts), p=pp)

                  # Select feature corresponding to chosen component
                  idx = np.arange((len(features)))
                  features = features[idx, opt_sel].reshape([-1, 1])

                  # Clip feature values to [-0.99, 0.99]
                  features = np.clip(features, -.99, .99)

                  # One-hot encode the chosen component
                  probs_onehot = np.zeros_like(probs)
                  probs_onehot[np.arange(len(probs)), opt_sel] = 1

                  # Re-order one-hot columns by frequency (so the most common components come first)
                  re_ordered_phot = np.zeros_like(probs_onehot)
                  col_sums = probs_onehot.sum(axis=0)
                  n = probs_onehot.shape[1]
                  largest_indices = np.argsort(-1*col_sums)[:n]
                  self.ordering.append(largest_indices)
                  for id,val in enumerate(largest_indices):
                      re_ordered_phot[:,id] = probs_onehot[:,val]

                  # Add continuous feature + reordered component encoding to values
                  values += [features, re_ordered_phot]

                else:
                  # If column *is* a general column, just normalize linearly to [-1,1]
                  self.ordering.append(None)

                  if id_ in self.non_categorical_columns:
                    # Adjust min/max slightly for stability
                    info['min'] = -1e-3
                    info['max'] = info['max'] + 1e-3

                  # Normalize to [-1,1]
                  current = (current - (info['min'])) / (info['max'] - info['min'])
                  current = current * 2 - 1
                  current = current.reshape([-1, 1])

                  # Append transformed values
                  values.append(current)

            # Case 2: Mixed column
            elif info['type'] == "mixed":
                # Extract parameters from gm1 (BayesianGaussianMixture) (trained on all values)
                means_0 = self.model[id_][0].means_.reshape([-1])
                stds_0 = np.sqrt(self.model[id_][0].covariances_).reshape([-1])

                # Identify mixture components that correspond to modal (placeholder) values
                zero_std_list = []
                means_needed = []
                stds_needed = []

                for mode in info['modal']:
                    if mode!=-9999999: # ignore placeholder modal
                        dist = []
                        for idx,val in enumerate(list(means_0.flatten())):
                            dist.append(abs(mode-val))
                        index_min = np.argmin(np.array(dist))
                        zero_std_list.append(index_min)
                    else: continue

                for idx in zero_std_list:
                    means_needed.append(means_0[idx])
                    stds_needed.append(stds_0[idx])

                # Calculate normalized distances for modal values
                mode_vals = []
                for i,j,k in zip(info['modal'],means_needed,stds_needed):
                    this_val  = np.abs(i - j) / (4*k)
                    mode_vals.append(this_val)

                # Add zero if modal contains -9999999
                if -9999999 in info["modal"]:
                    mode_vals.append(0)

                # Filter out placeholder values
                current = current.reshape([-1, 1])
                filter_arr = self.filter_arr[mixed_counter]
                current = current[filter_arr]

                # Extract parameters from gm2 (trained on valid values only)
                means = self.model[id_][1].means_.reshape((1, self.n_clusters))
                stds = np.sqrt(self.model[id_][1].covariances_).reshape((1, self.n_clusters))

                # Normalize continuous part
                features = np.empty(shape=(len(current),self.n_clusters))
                if ispositive == True:
                    if id_ in positive_list:
                        features = np.abs(current - means) / (4 * stds)
                else:
                    features = (current - means) / (4 * stds)

                # Mixture component probabilities
                probs = self.model[id_][1].predict_proba(current.reshape([-1, 1]))

                # Keep only active components
                n_opts = sum(self.components[id_]) # 8
                features = features[:, self.components[id_]]
                probs = probs[:, self.components[id_]]

                # Sample component index per row
                opt_sel = np.zeros(len(current), dtype='int')
                for i in range(len(current)):
                    pp = probs[i] + 1e-6
                    pp = pp / sum(pp)
                    opt_sel[i] = np.random.choice(np.arange(n_opts), p=pp)

                # Pick selected feature values
                idx = np.arange((len(features)))
                features = features[idx, opt_sel].reshape([-1, 1])
                features = np.clip(features, -.99, .99)

                # One-hot encode sampled component
                probs_onehot = np.zeros_like(probs)
                probs_onehot[np.arange(len(probs)), opt_sel] = 1

                # Build an extended vector: [modal one-hots + component one-hots]
                extra_bits = np.zeros([len(current), len(info['modal'])])
                temp_probs_onehot = np.concatenate([extra_bits,probs_onehot], axis = 1)

                # Build final encoding for the entire dataset length
                final = np.zeros([len(data), 1 + probs_onehot.shape[1] + len(info['modal'])])
                features_curser = 0
                for idx, val in enumerate(data[:, id_]):
                    if val in info['modal']:
                        # placeholder case
                        category_ = list(map(info['modal'].index, [val]))[0]
                        final[idx, 0] = mode_vals[category_] # normalized placeholder
                        final[idx, (category_+1)] = 1        # one-hot placeholder

                    else:
                        # valid continuous case
                        final[idx, 0] = features[features_curser]
                        final[idx, (1+len(info['modal'])):] = temp_probs_onehot[features_curser][len(info['modal']):]
                        features_curser = features_curser + 1

                # Reorder one-hots by frequency (most common categories/components first)
                just_onehot = final[:,1:]
                re_ordered_jhot= np.zeros_like(just_onehot)
                n = just_onehot.shape[1]
                col_sums = just_onehot.sum(axis=0)
                largest_indices = np.argsort(-1*col_sums)[:n]
                self.ordering.append(largest_indices)
                for id,val in enumerate(largest_indices):
                      re_ordered_jhot[:,id] = just_onehot[:,val]

                # Split into continuous feature + reordered one-hots
                final_features = final[:,0].reshape([-1, 1])
                values += [final_features, re_ordered_jhot]
                mixed_counter = mixed_counter + 1

            # Case 3: Categorical column
            else:
                self.ordering.append(None)

                # Convert categorical to one-hot
                col_t = np.zeros([len(data), info['size']])
                idx = list(map(info['i2s'].index, current))
                col_t[np.arange(len(data)), idx] = 1
                values.append(col_t)

        # Concatenate all encodings into final transformed dataset
        return np.concatenate(values, axis=1)

    def inverse_transform(self, data):
        # Initialize output matrix (same #rows as data, same #cols as original features)
        data_t = np.zeros([len(data), len(self.meta)])

        # Track invalid rows (values outside expected range after decoding)
        invalid_ids = []

        # Start index in the encoded data (tracks where each feature’s encoding starts)
        st = 0

        # Loop through each column’s metadata
        for id_, info in enumerate(self.meta):
            # Case 1: Continuous column
            if info['type'] == "continuous":

                # If not a "general" column, then it was modeled with GMM
                if id_ not in self.general_columns:
                  #  Extract encoded pieces:
                  u = data[:, st]   # scaled continuous value (tanh space)
                  v = data[:, st + 1:st + 1 + np.sum(self.components[id_])]  # mixture softmax part

                  # Reorder mixture probs back to original component order
                  order = self.ordering[id_]
                  v_re_ordered = np.zeros_like(v)
                  for id, val in enumerate(order):
                      v_re_ordered[:,val] = v[:,id]
                  v = v_re_ordered

                  # Clip u to [-1,1] range (safe bounds)
                  u = np.clip(u, -1, 1)

                  # Expand v back to full number of clusters (inactive comps = -100 placeholder)
                  v_t = np.ones((data.shape[0], self.n_clusters)) * -100
                  v_t[:, self.components[id_]] = v
                  v = v_t

                   # Update index pointer (1 for u + #active comps for v)
                  st += 1 + np.sum(self.components[id_])

                  # Recover means/stds of GMM components
                  means = self.model[id_].means_.reshape([-1])
                  stds = np.sqrt(self.model[id_].covariances_).reshape([-1])

                  # Pick most likely Gaussian component per row
                  p_argmax = np.argmax(v, axis=1)
                  std_t = stds[p_argmax]
                  mean_t = means[p_argmax]

                  # Reconstruct original value from scaled u
                  tmp = u * 4 * std_t + mean_t

                  # Mark invalid rows if outside min/max bounds
                  for idx,val in enumerate(tmp):
                     if (val < info["min"]) | (val > info['max']):
                         invalid_ids.append(idx)

                  # If it’s an integer column, round values
                  if id_ in self.non_categorical_columns:
                    tmp = np.round(tmp)

                  # Save reconstructed values
                  data_t[:, id_] = tmp

                # If it's a "general" continuous column (no GMM, just scaled linearly)
                else:
                  u = data[:, st]
                  u = (u + 1) / 2   # scale from [-1,1] back to [0,1]
                  u = np.clip(u, 0, 1)  # ensure within bounds
                  u = u * (info['max'] - info['min']) + info['min']  # scale to original range

                  if id_ in self.non_categorical_columns:
                    data_t[:, id_] = np.round(u)
                  else: data_t[:, id_] = u

                  st += 1

            # Case 2: Mixed column
            elif info['type'] == "mixed":
                # Extract u (scaled continuous) and v (categorical+mixture one-hot)
                u = data[:, st]
                full_v = data[:,(st+1):(st+1)+len(info['modal'])+np.sum(self.components[id_])]

                # Reorder one-hot back to original order
                order = self.ordering[id_]
                full_v_re_ordered = np.zeros_like(full_v)
                for id, val in enumerate(order):
                    full_v_re_ordered[:,val] = full_v[:,id]
                full_v = full_v_re_ordered

                # Split into placeholder one-hot (modal values) vs mixture one-hot
                mixed_v = full_v[:,:len(info['modal'])]
                v = full_v[:,-np.sum(self.components[id_]):]

                # Clip u, expand v back to full GMM cluster dimension
                u = np.clip(u, -1, 1)
                v_t = np.ones((data.shape[0], self.n_clusters)) * -100
                v_t[:, self.components[id_]] = v

                # Concatenate placeholder + Gaussian mixture one-hot
                v = np.concatenate([mixed_v,v_t], axis=1)

                # Update pointer
                st += 1 + np.sum(self.components[id_]) + len(info['modal'])

                # Retrieve means/stds for Gaussian components
                means = self.model[id_][1].means_.reshape([-1])
                stds = np.sqrt(self.model[id_][1].covariances_).reshape([-1])

                # Find most likely component index per row
                p_argmax = np.argmax(v, axis=1)

                # Allocate array for reconstructed values
                result = np.zeros_like(u)

                for idx in range(len(data)):
                    # Case A: belongs to placeholder/modal category
                    if p_argmax[idx] < len(info['modal']):
                        argmax_value = p_argmax[idx]
                        result[idx] = float(list(map(info['modal'].__getitem__, [argmax_value]))[0])

                    # Case B: belongs to Gaussian continuous part
                    else:
                        std_t = stds[(p_argmax[idx]-len(info['modal']))]
                        mean_t = means[(p_argmax[idx]-len(info['modal']))]
                        result[idx] = u[idx] * 4 * std_t + mean_t

                # Mark invalid rows if outside min/max
                for idx,val in enumerate(result):
                     if (val < info["min"]) | (val > info['max']):
                         invalid_ids.append(idx)

                # Save reconstructed column
                data_t[:, id_] = result

            # Case 3: Categorical column
            else:
                # Extract one-hot encoded part for this column
                current = data[:, st:st + info['size']]
                st += info['size']

                # Convert one-hot back to integer index
                idx = np.argmax(current, axis=1)

                # Map indices back to original string labels using i2s
                data_t[:, id_] = list(map(info['i2s'].__getitem__, idx))

        # After all columns reconstructed:
        # Collect unique invalid row indices
        invalid_ids = np.unique(np.array(invalid_ids))

        # All row indices
        all_ids = np.arange(0,len(data))

        # Keep only valid row indices
        valid_ids = list(set(all_ids) - set(invalid_ids))

        # Return reconstructed data (only valid rows) and # of invalid rows dropped
        return data_t[valid_ids],len(invalid_ids)

# A helper class to reshape 1D tabular data into 2D image format (and back)
class ImageTransformer():
    # Save the "side length" of the square image (e.g., side=28 28x28 image)
    def __init__(self, side):
        self.height = side

    # Convert flat vectors into square image tensors.
    def transform(self, data):
        # If the vector length is smaller than side*side, add zero-padding
        if self.height * self.height > len(data[0]):
            # Create a zero padding tensor for each row
            padding = torch.zeros((len(data), self.height * self.height - len(data[0]))).to(data.device)

            # Concatenate original data with padding along feature dimension
            data = torch.cat([data, padding], axis=1)

        # Reshape into image format: (batch_size, channels=1, height, width)
        return data.view(-1, 1, self.height, self.height)

    # Convert image tensors back into flat vectors
    def inverse_transform(self, data):
        # Flatten the image back into (batch_size, height*height)
        data = data.view(-1, self.height * self.height)
        return data


# ___________________________________________________
# Class to define each model/component of CTAB-GAN+
class Classifier(Module):
    def __init__(self,input_dim, dis_dims,st_ed):
        """
        input_dim : int
            Total dimension of the input vector.
        dis_dims : list[int]
            Hidden layer sizes for the classifier.
        st_ed : tuple(start_idx, end_idx)
            Index range in the input that corresponds to the target column(s)
            we want to predict.
        """
        super(Classifier,self).__init__()

        # Remaining feature dimension after removing the label part
        dim = input_dim-(st_ed[1]-st_ed[0])

        # List to store the sequence of layers
        seq = []

        # Save the target column index range
        self.str_end = st_ed

        # Build hidden layers based on dis_dims
        for item in list(dis_dims):
            seq += [

                  Linear(dim, item),     # Fully connected layer
                LeakyReLU(0.2),        # LeakyReLU activation with slope 0.2
                Dropout(0.5)           # Dropout for regularization (50% drop rate)
            ]
            dim = item # update current dimension for the next layer

        # Define the output layer depending on target column size
        if (st_ed[1] - st_ed[0]) == 1:
            # Case 1: Single continuous value, regression (no activation)
            seq += [Linear(dim, 1)]

        elif (st_ed[1] - st_ed[0]) == 2:
            # Case 2: Binary classification, 1 output with sigmoid
            seq += [Linear(dim, 1),Sigmoid()]

        else:
            # Case 3: Multi-class classification, softmax handled outside loss
            seq += [Linear(dim, (st_ed[1] - st_ed[0]))]

        # Convert list into a Sequential model
        self.seq = Sequential(*seq)

    def forward(self, input):
        """
        input : Tensor of shape (batch_size, input_dim)
            Full feature vector containing both label columns and other features.

        Returns:
            prediction : Tensor
                Model's predicted values.
            label : Tensor
                Ground-truth labels extracted from input.
        """
        label = None

        # Extract label from input based on st_ed
        if (self.str_end[1] - self.str_end[0]) == 1:
            # For regression (single value), take slice directly
            label = input[:, self.str_end[0]:self.str_end[1]]
        else:
            # For classification, take argmax across label columns
            label = torch.argmax(input[:, self.str_end[0]:self.str_end[1]], axis = -1)

        # Remove label columns from input, keep only features for training
        new_imp = torch.cat((input[:,:self.str_end[0]], input[:, self.str_end[1]:]), 1)# left + right parts

        # Forward through classifier depending on task type
        if ((self.str_end[1] - self.str_end[0]) == 2) | ((self.str_end[1] - self.str_end[0]) == 1):
            # Regression or binary classification
            return self.seq(new_imp).view(-1), label
        else:
            # Multi-class classification
            return self.seq(new_imp), label

def apply_activate(data, output_info):
    """
    Apply the appropriate activation function to different sections of the generator output.
    data : Tensor
        Raw generator outputs.
    output_info : list of tuples
        Contains information for each feature segment:
        (dimension, activation type, optional flag)
        Example: [(1, 'tanh', 'yes_g'), (5, 'softmax')]
    Returns:
        Tensor with activations applied, concatenated.
    """
    data_t = []  # List to store activated segments
    st = 0       # Start index in the data

    for item in output_info:
        if item[1] == 'tanh':
            # Apply tanh activation to continuous segments
            ed = st + item[0] # End index of this segment
            data_t.append(torch.tanh(data[:, st:ed]))
            st = ed # Move start pointer to next segment

        elif item[1] == 'softmax':
            # Apply Gumbel-softmax activation to categorical segments
            ed = st + item[0]
            data_t.append(F.gumbel_softmax(data[:, st:ed], tau = 0.2))
            st = ed

    # Concatenate all activated segments along the feature dimension
    return torch.cat(data_t, dim = 1)

def get_st_ed(target_col_index,output_info):
    """
    Compute the start and end positions in the flattened generator output
    corresponding to the target column index.
    target_col_index : int
        Index of the target column (among all columns).
    output_info : list of tuples
        Metadata about each output segment.
    Returns:
        (start, end) indices in the generator output for the target column.
    """
    st = 0  # start pointer
    c = 0   # column counter
    tc = 0  # segment counter

    for item in output_info:
        if c == target_col_index:
            break
        if item[1] == 'tanh':
            st += item[0]
            if item[2] == 'yes_g': # only count general continuous columns
                c += 1
        elif item[1] == 'softmax':
            st += item[0]
            c += 1
        tc += 1 # move to next segment

    ed= st + output_info[tc][0] # end index = start + segment length
    return (st, ed)

def random_choice_prob_index_sampling(probs,col_idx):
    """
    For each row, sample a categorical value based on provided probability distributions.
    probs : 2D array
        Each row contains probabilities for each category.
    col_idx : array-like
        Indices of the rows to sample from.
    Returns:
        Array of sampled indices (one per selected row).
    """
    option_list = []
    for i in col_idx:
        pp = probs[i] # probability distribution for row i
        option_list.append(np.random.choice(np.arange(len(probs[i])), p = pp))

    # Return as array with same shape as col_idx
    return np.array(option_list).reshape(col_idx.shape)

def random_choice_prob_index(a, axis=1):
    """
    Vectorized sampling from categorical distributions using cumulative sums.
    a : np.array
        Probabilities along the specified axis.
    axis : int
        Axis along which to sample (default 1 for row-wise).
    Returns:
        Array of sampled indices.
    """
    r = np.expand_dims(np.random.rand(a.shape[1 - axis]), axis = axis)  # random threshold
    return (a.cumsum(axis = axis) > r).argmax(axis = axis)  # pick first index where cumulative sum > random number

def maximum_interval(output_info):
    """
    Find the maximum segment length among all output segments.
    output_info : list of tuples
        Metadata for each output segment.
    Returns:
        int : the largest segment dimension
    """
    max_interval = 0
    for item in output_info:
        max_interval = max(max_interval, item[0]) # keep track of largest segment
    return max_interval

class Cond(object):
    """
    Conditional sampler for categorical features.
    This class extracts categorical (softmax) columns from the generator output,
    calculates probabilities, and provides sampling methods for training or generation.
    """
    def __init__(self, data, output_info):
        """
        Initialize conditional information from data.
        data : np.array
            Generated data (output from generator) with both continuous and categorical features.
        output_info : list of tuples
            Metadata describing each feature segment: (dimension, activation_type, optional_flag)
        """
        self.model = []  # Stores argmax indices of categorical columns
        st = 0           # Start index in flattened data
        counter = 0      # Counts number of categorical columns

        # Step 1: Extract argmax (category index) for softmax features
        for item in output_info:

            if item[1] == 'tanh':
                st += item[0] # skip continuous features
                continue
            elif item[1] == 'softmax':
                ed = st + item[0]
                counter += 1
                # Store the argmax of each row (most likely category)
                self.model.append(np.argmax(data[:, st:ed], axis = -1))
                st = ed

        # Step 2: Prepare probability vectors and intervals
        self.interval = []  # Stores start index and size of each categorical column in flattened vec
        self.n_col = 0      # Number of categorical columns
        self.n_opt = 0      # Total number of options across all categorical columns
        st = 0
        self.p = np.zeros((counter, maximum_interval(output_info)))  # Probabilities for training
        self.p_sampling = []  # Probabilities for generation

        # Loop again to compute probability distributions
        for item in output_info:
            if item[1] == 'tanh':
                st += item[0]
                continue
            elif item[1] == 'softmax':
                ed = st + item[0]

                # Compute probability distributions
                tmp = np.sum(data[:, st:ed], axis=0)  # sum across batch (training probabilities)
                tmp_sampling = np.sum(data[:, st:ed], axis=0)  # copy for sampling
                tmp = np.log(tmp + 1)  # log scale to avoid zeros
                tmp = tmp / np.sum(tmp)  # normalize
                tmp_sampling = tmp_sampling / np.sum(tmp_sampling)  # normalize

                self.p_sampling.append(tmp_sampling)  # for generator sampling
                self.p[self.n_col, :item[0]] = tmp     # store for training

                # Record interval (start position and length of categorical segment)
                self.interval.append((self.n_opt, item[0]))

                # Update counters
                self.n_opt += item[0]
                self.n_col += 1
                st = ed

        self.interval = np.asarray(self.interval) # convert to np.array for indexing

    # Sampling functions
    def sample_train(self, batch):
        """
        Sample conditional vectors for training.
        batch : int
            Number of samples to generate.
        Returns:
            vec : np.array
                One-hot vectors for selected categorical columns.
            mask : np.array
                Indicates which column was selected per sample.
            idx : np.array
                Indices of selected categorical columns.
            opt1prime : np.array
                Sampled category indices for selected columns.
        """
        if self.n_col == 0:
            return None # no categorical columns

        idx = np.random.choice(np.arange(self.n_col), batch) # randomly select a categorical column per sample

        vec = np.zeros((batch, self.n_opt), dtype='float32')  # one-hot output
        mask = np.zeros((batch, self.n_col), dtype='float32') # mask to indicate which column is chosen
        mask[np.arange(batch), idx] = 1

        # Sample category index for selected column using probability distribution
        opt1prime = random_choice_prob_index(self.p[idx])

        # Fill in one-hot vector for each sample
        for i in np.arange(batch):
            vec[i, self.interval[idx[i], 0] + opt1prime[i]] = 1

        return vec, mask, idx, opt1prime

    def sample(self, batch):
        """
        Sample conditional vectors for generation (no mask returned).
        batch : int
            Number of samples to generate.
        Returns:
            vec : np.array
                One-hot vectors for selected categorical columns.
        """
        if self.n_col == 0:
            return None

        idx = np.random.choice(np.arange(self.n_col), batch) # randomly select a categorical column per sample

        vec = np.zeros((batch, self.n_opt), dtype='float32') # one-hot output

        # Sample category indices for generator using normalized probabilities
        opt1prime = random_choice_prob_index_sampling(self.p_sampling,idx)

        # Fill in one-hot vector for each sample
        for i in np.arange(batch):
            vec[i, self.interval[idx[i], 0] + opt1prime[i]] = 1

        return vec

def cond_loss(data, output_info, c, m):
    """
    Compute the conditional loss (cross-entropy) for categorical columns.
    data : Tensor
        Generator outputs (after activation) of shape (batch_size, total_features)
    output_info : list of tuples
        Metadata describing each feature segment: (dimension, activation_type, optional_flag)
    c : Tensor
        Conditional one-hot vectors used as ground truth (batch_size, total_features)
    m : Tensor
        Mask indicating which categorical columns to include in the loss (batch_size, n_cat_columns)
    Returns:
        Scalar tensor representing the conditional loss.
    """
    loss = []     # List to store per-column losses
    st = 0        # start index for generator output
    st_c = 0      # start index for conditional ground-truth

    # Loop through all output segments
    for item in output_info:
        if item[1] == 'tanh':
            # Skip continuous segments
            st += item[0]
            continue

        elif item[1] == 'softmax':
            # Process categorical segments
            ed = st + item[0]      # end index for generator output
            ed_c = st_c + item[0]  # end index for conditional vector

            # Compute cross-entropy loss for this categorical column
            tmp = F.cross_entropy(
                data[:, st:ed],                        # predicted logits for this column
                torch.argmax(c[:, st_c:ed_c], dim=1),  # target category indices
                reduction='none'                       # keep loss per sample (not averaged)
            )

            # Append loss for this column
            loss.append(tmp)

            # Move start pointers to next segment
            st = ed
            st_c = ed_c

    # Stack all column losses along a new dimension (batch_size, n_cat_columns)
    loss = torch.stack(loss, dim=1)

    # Apply mask, sum over columns, then average over batch
    return (loss * m).sum() / data.size()[0]

class Sampler(object):
    """
    Sampler for dataset rows, optionally conditioned on specific categorical features.
    """
    def __init__(self, data, output_info):
        """
        Initialize the Sampler.
        data : np.array
            Dataset to sample from (rows = samples, columns = features).
        output_info : list of tuples
            Metadata describing each feature segment: (dimension, activation_type, optional_flag)
        """
        super(Sampler, self).__init__()
        self.data = data        # store full dataset
        self.model = []         # list to store indices of non-zero entries per categorical option
        self.n = len(data)      # total number of rows (samples)
        st = 0                  # start index in flattened data

        # Loop over all segments of output_info
        for item in output_info:
            if item[1] == 'tanh':
                st += item[0] # skip continuous features
                continue
            elif item[1] == 'softmax':
                ed = st + item[0]  # end index of this categorical segment
                tmp = []           # temporary list to store row indices for each category

                # Loop through each category in this softmax segment
                for j in range(item[0]):
                    # np.nonzero returns row indices where this category is active (one-hot = 1)
                    tmp.append(np.nonzero(data[:, st + j])[0])

                # Append the list of row indices per category for this categorical column
                self.model.append(tmp)
                st = ed # move start pointer to next segment

    def sample(self, n, col, opt):
        """
        Sample n rows from the dataset, optionally conditioned on categorical columns.
        n : int
            Number of samples to draw.
        col : list or None
            List of categorical column indices to condition on. If None, sample randomly.
        opt : list or None
            List of categorical option indices for each column in `col`. Must match length of `col`.
        Returns:
            np.array : sampled rows from self.data
        """
        if col is None:
            # Random sampling across all rows
            idx = np.random.choice(np.arange(self.n), n)
            return self.data[idx]

        # Conditional sampling based on specified categorical columns and options
        idx = []
        for c, o in zip(col, opt):
            # Pick a random row index where the selected categorical column 'c' has option 'o'
            idx.append(np.random.choice(self.model[c][o]))

        # Return sampled rows
        return self.data[idx]

class Discriminator(Module):
    """
    Discriminator network for a GAN.
    It tries to distinguish between real and generated (fake) data.
    """
    def __init__(self, side, layers):
        super(Discriminator, self).__init__()
        self.side = side                 # store "side" (may represent input shape like image side length)
        info = len(layers) - 2           # index up to which we extract intermediate features (exclude last 2 layers)

        # full discriminator model (all layers)
        self.seq = Sequential(*layers)

        # partial discriminator (all but the last 2 layers), useful for feature extraction
        self.seq_info = Sequential(*layers[:info])

    def forward(self, input):
        """
        Forward pass of the discriminator.
        input : torch.Tensor
            Input data (real or generated).
        Returns:
            - output from full network (real/fake probability)
            - output from intermediate layers (feature representation)
        """
        return (self.seq(input)), self.seq_info(input)

class Generator(Module):
    """
    Generator network for a GAN.
    It learns to produce synthetic data that resembles real data.
    """
    def __init__(self, side, layers):
        super(Generator, self).__init__()
        self.side = side                  # store "side" (may represent output shape like image side length)

        # generator model composed of provided layers
        self.seq = Sequential(*layers)

    def forward(self, input_):
        """
        Forward pass of the generator.
        input_ : torch.Tensor
            Input noise or latent vector.
        Returns:
            Generated synthetic sample.
        """
        return self.seq(input_)

def determine_layers_disc(side, num_channels):
    """
    Build the discriminator layers for a CNN-based GAN.
    Args:
        side (int): Image side length (must be between 4 and 64).
        num_channels (int): Number of feature channels for the first conv layer.
    Returns:
        layers_D (list): A list of PyTorch layers forming the discriminator.
    """
    assert side >= 4 and side <= 64 # ensure valid image size

    # Start with input dims: grayscale (1 channel), and first conv feature map
    layer_dims = [(1, side), (num_channels, side // 2)]

    # Add more layers: double channels, halve spatial size until small enough
    while layer_dims[-1][1] > 3 and len(layer_dims) < 4:
        layer_dims.append((layer_dims[-1][0] * 2, layer_dims[-1][1] // 2))

    # Compute shapes for LayerNorm for each stage
    layerNorms = []
    num_c = num_channels
    num_s = side / 2
    for l in range(len(layer_dims) - 1):
        layerNorms.append([int(num_c), int(num_s), int(num_s)])
        num_c = num_c * 2
        num_s = num_s / 2

    layers_D = []

    # Build conv → layernorm → leaky relu blocks
    for prev, curr, ln in zip(layer_dims, layer_dims[1:], layerNorms):
        layers_D += [
            Conv2d(prev[0], curr[0], 4, 2, 1, bias=False),  # downsample by stride 2
            LayerNorm(ln),                                 # normalize per layer
            LeakyReLU(0.2, inplace=True),                  # leaky relu activation
        ]

    # Final conv layer outputs a single value (real/fake score)
    layers_D += [
        Conv2d(layer_dims[-1][0], 1, layer_dims[-1][1], 1, 0),
        ReLU(True) # final nonlinearity
    ]

    return layers_D

def determine_layers_gen(side, random_dim, num_channels):
    """
    Build the generator layers for a CNN-based GAN.
    Args:
        side (int): Target image side length.
        random_dim (int): Dimension of latent noise vector (input).
        num_channels (int): Number of channels in the second-to-last layer.
    Returns:
        layers_G (list): A list of PyTorch layers forming the generator.
    """
    assert side >= 4 and side <= 64

    # Define feature map dimensions from input to output
    layer_dims = [(1, side), (num_channels, side // 2)]
    while layer_dims[-1][1] > 3 and len(layer_dims) < 4:
        layer_dims.append((layer_dims[-1][0] * 2, layer_dims[-1][1] // 2))

    # Compute shapes for LayerNorm for each stage (reverse of discriminator)
    layerNorms = []
    num_c = num_channels * (2 ** (len(layer_dims) - 2))
    num_s = int(side / (2 ** (len(layer_dims) - 1)))
    for l in range(len(layer_dims) - 1):
        layerNorms.append([int(num_c), int(num_s), int(num_s)])
        num_c = num_c / 2
        num_s = num_s * 2

    # First layer: project latent vector into a feature map
    layers_G = [
        ConvTranspose2d(
            random_dim,
            layer_dims[-1][0],
            layer_dims[-1][1], 1, 0,
            output_padding=0,
            bias=False
        )
    ]

    # Add transposed conv blocks (upsampling)
    for prev, curr, ln in zip(reversed(layer_dims), reversed(layer_dims[:-1]), layerNorms):
        layers_G += [
            LayerNorm(ln),    # normalize features
            ReLU(True),       # activation
            ConvTranspose2d(  # upsample by stride 2
                prev[0],
                curr[0], 4, 2, 1,
                output_padding=0,
                bias=True
            )
        ]
    return layers_G

def slerp(val, low, high):
    """
    Perform spherical linear interpolation (slerp) between two vectors.
    Often used for smooth latent space interpolation in GANs.
    Args:
        val (Tensor): interpolation factor (0 → low, 1 → high).
        low (Tensor): first latent vector.
        high (Tensor): second latent vector.
    Returns:
        res (Tensor): interpolated vector.
    """
    # Normalize both vectors
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)

    # Compute angle between vectors
    omega = torch.acos((low_norm*high_norm).sum(1)).view(val.size(0), 1)
    so = torch.sin(omega)

    # Spherical interpolation formula
    res = (torch.sin((1.0-val)*omega)/so)*low + (torch.sin(val*omega)/so) * high

    return res

def calc_gradient_penalty_slerp(netD, real_data, fake_data, transformer, device='cpu', lambda_=10):
    """
    Compute gradient penalty for WGAN-GP using slerp interpolation.
    Args:
        netD: Discriminator network.
        real_data: batch of real samples.
        fake_data: batch of fake (generated) samples.
        transformer: data transformer (reshaping input if needed).
        device: device ('cpu' or 'cuda').
        lambda_: gradient penalty coefficient.
    Returns:
        gradient_penalty (Tensor): scalar penalty value.
    """
    batchsize = real_data.shape[0]

    # Random interpolation factor
    alpha = torch.rand(batchsize, 1,  device=device)

    # Interpolate between real and fake samples using slerp
    interpolates = slerp(alpha, real_data, fake_data)
    interpolates = interpolates.to(device)
    interpolates = transformer.transform(interpolates)

    # Require gradients for interpolated samples
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    # Pass interpolates through discriminator
    disc_interpolates,_ = netD(interpolates)

    # Compute gradients of discriminator output w.r.t interpolates
    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

    # Compute L2 norm of gradients
    gradients_norm = gradients.norm(2, dim=1)

    # Gradient penalty: encourage ||grad|| ≈ 1
    gradient_penalty = ((gradients_norm - 1) ** 2).mean() * lambda_

    return gradient_penalty

def weights_init(m):
    """
    Custom weight initialization for GAN layers.
    Uses DCGAN-style initialization.
    """
    classname = m.__class__.__name__

    # For convolution layers: normal distribution with mean=0, std=0.02
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)

    # For batch norm layers: normal for weights (mean=1, std=0.02), constant for bias (0)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0)

class CTABGANSynthesizer:
    def __init__(self,
                 class_dim=(256, 256, 256, 256),  # hidden layer sizes for classifier network
                 random_dim=100,                  # dimension of random noise vector z
                 num_channels=64,                 # number of channels used in CNN layers
                 l2scale=1e-5,                    # weight decay (L2 regularization)
                 batch_size=500,                  # training batch size
                 epochs=1000):                    # number of training epochs

        self.private = False                      # Flag for Differential Privacy training (disabled by default)
        self.micro_batch_size = batch_size        # for DP training (splits batch into smaller micro-batches)

        # Hyperparameters for Differential Privacy (only used if self.private=True)
        # clip_coeff and sigma are the hyper-parameters for injecting noise in gradients
        self.clip_coeff = 1                       # gradient clipping coefficient
        self.sigma = 1.02                         # standard deviation for Gaussian noise added to gradients

        # target delta is prefixed
        self.target_delta = 1e-5                  # target privacy budget parameter δ

        # Store model hyperparameters
        self.random_dim = random_dim
        self.class_dim = class_dim
        self.num_channels = num_channels
        self.dside = None                         # discriminator input side length (calculated later)
        self.gside = None                         # generator input side length (calculated later)
        self.l2scale = l2scale
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # use GPU if available

    def fit(self, train_data=pd.DataFrame, categorical=[], mixed={}, general=[], non_categorical=[], type={}):
        """
        Fit CTAB-GAN to training data.
        Handles preprocessing, builds networks, and runs adversarial training.
        """
        # 1. Identify target column if supervised task
        problem_type = None
        target_index=None
        if type:                                   # e.g. {"classification": "target_col"}
            problem_type = list(type.keys())[0]    # task type (classification/regression)
            if problem_type:
                target_index = train_data.columns.get_loc(type[problem_type]) # get column index of target

        # 2. Data Transformation
        # Convert categorical + continuous columns into normalized + encoded tensors
        self.transformer = DataTransformer(
            train_data=train_data,
            categorical_list=categorical,
            mixed_dict=mixed,
            general_list=general,
            non_categorical_list=non_categorical
        )

        self.transformer.fit()                                    # fit transformer (learn encodings, scalers, etc.)
        train_data = self.transformer.transform(train_data.values) # transform actual data

        # Sampler object for real data
        data_sampler = Sampler(train_data, self.transformer.output_info)
        data_dim = self.transformer.output_dim # total transformed data dimension

        # Conditional vector generator (for conditioning GAN on categorical values)
        self.cond_generator = Cond(train_data, self.transformer.output_info)

        # 3. Determine network input/output sizes
        sides = [4, 8, 16, 24, 32, 64]  # candidate square image sizes

        # Discriminator "image" side length (depends on data + conditional vector size)
        col_size_d = data_dim + self.cond_generator.n_opt
        for i in sides:
            if i * i >= col_size_d:
                self.dside = i
                break

        # Generator "image" side length (depends only on data dimension)
        col_size_g = data_dim
        for i in sides:
            if i * i >= col_size_g:
                self.gside = i
                break

        # 4. Build Generator and Discriminator
        layers_G = determine_layers_gen(self.gside, self.random_dim+self.cond_generator.n_opt, self.num_channels)
        layers_D = determine_layers_disc(self.dside, self.num_channels)

        self.generator = Generator(self.gside, layers_G).to(self.device)
        discriminator = Discriminator(self.dside, layers_D).to(self.device)

        # Optimizers
        optimizer_params = dict(lr=2e-4, betas=(0.5, 0.9), eps=1e-3, weight_decay=self.l2scale)
        optimizerG = Adam(self.generator.parameters(), **optimizer_params)
        optimizerD = Adam(discriminator.parameters(), **optimizer_params)

        # 5. (Optional) Add auxiliary classifier if supervised task
        st_ed = None
        classifier=None
        optimizerC= None
        if target_index != None: # supervised mode
            st_ed= get_st_ed(target_index,self.transformer.output_info)
            classifier = Classifier(data_dim,self.class_dim,st_ed).to(self.device)
            optimizerC = optim.Adam(classifier.parameters(),**optimizer_params)

        # Initialize model weights
        self.generator.apply(weights_init)
        discriminator.apply(weights_init)

        # Transformers to reshape tabular data <-> image-like format
        self.Gtransformer = ImageTransformer(self.gside)
        self.Dtransformer = ImageTransformer(self.dside)

        # 6. Training Loop
        epsilon = 0
        epoch = 0
        steps = 0
        ci = 1 # number of critic iterations per generator step

        steps_per_epoch = max(1, len(train_data) // self.batch_size)

        for i in tqdm(range(self.epochs)):         # loop over epochs
            for id_ in range(steps_per_epoch):     # loop over mini-batches
                # Train Discriminator
                for _ in range(ci): # run multiple critic updates
                    noisez = torch.randn(self.batch_size, self.random_dim, device=self.device)  # sample random noise
                    condvec = self.cond_generator.sample_train(self.batch_size)                 # sample conditional vector

                    c, m, col, opt = condvec
                    c = torch.from_numpy(c).to(self.device)
                    m = torch.from_numpy(m).to(self.device)

                    # Append conditional vector to noise
                    noisez = torch.cat([noisez, c], dim=1)
                    noisez =  noisez.view(self.batch_size,self.random_dim+self.cond_generator.n_opt,1,1)

                    # Sample real data conditioned on categorical values
                    perm = np.arange(self.batch_size)
                    np.random.shuffle(perm)
                    real = data_sampler.sample(self.batch_size, col[perm], opt[perm])
                    c_perm = c[perm]

                    real = torch.from_numpy(real.astype('float32')).to(self.device)

                    # Generate fake samples
                    fake = self.generator(noisez)
                    faket = self.Gtransformer.inverse_transform(fake) # flatten image -> tabular
                    fakeact = apply_activate(faket, self.transformer.output_info)

                    # Concatenate with conditional vector
                    fake_cat = torch.cat([fakeact, c], dim=1)
                    real_cat = torch.cat([real, c_perm], dim=1)

                    # Transform into "image-like" input for discriminator
                    real_cat_d = self.Dtransformer.transform(real_cat)
                    fake_cat_d = self.Dtransformer.transform(fake_cat)

                    optimizerD.zero_grad()

                    # Forward pass on real samples
                    d_real,_ = discriminator(real_cat_d)

                    # If using Differential Privacy
                    # following block cliping gradients and add noises.
                    if self.private:
                        # Gradient clipping + Gaussian noise injection
                        clipped_grads = {
                            name: torch.zeros_like(param) for name, param in discriminator.named_parameters()}

                        for k in range(int(d_real.size(0) / self.micro_batch_size)):
                            err_micro = -1*d_real[k * self.micro_batch_size: (k + 1) * self.micro_batch_size].mean(0).view(1)
                            err_micro.backward(retain_graph=True)
                            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), self.clip_coeff)
                            for name, param in discriminator.named_parameters():
                                clipped_grads[name] += param.grad
                            discriminator.zero_grad()

                        for name, param in discriminator.named_parameters():
                            param.grad = (clipped_grads[name] + torch.FloatTensor(
                                clipped_grads[name].size()).normal_(0, self.sigma * self.clip_coeff).cuda()) / (
                                                     d_real.size(0) / self.micro_batch_size)

                        steps += 1

                    else:
                        # WGAN-GP objective for real data
                        d_real = -torch.mean(d_real)
                        d_real.backward()

                    # Forward pass on fake samples
                    d_fake,_ = discriminator(fake_cat_d)
                    d_fake = torch.mean(d_fake)
                    d_fake.backward()

                    # Gradient penalty for Lipschitz constraint
                    pen = calc_gradient_penalty_slerp(discriminator, real_cat, fake_cat,  self.Dtransformer , self.device)
                    pen.backward()

                    optimizerD.step()

                # (b) Train Generator
                noisez = torch.randn(self.batch_size, self.random_dim, device=self.device)
                condvec = self.cond_generator.sample_train(self.batch_size)

                c, m, col, opt = condvec
                c = torch.from_numpy(c).to(self.device)
                m = torch.from_numpy(m).to(self.device)

                noisez = torch.cat([noisez, c], dim=1)
                noisez =  noisez.view(self.batch_size,self.random_dim+self.cond_generator.n_opt,1,1)

                optimizerG.zero_grad()

                # Generate fake samples
                fake = self.generator(noisez)
                faket = self.Gtransformer.inverse_transform(fake)
                fakeact = apply_activate(faket, self.transformer.output_info)

                fake_cat = torch.cat([fakeact, c], dim=1)
                fake_cat = self.Dtransformer.transform(fake_cat)

                # Discriminator prediction
                y_fake, info_fake = discriminator(fake_cat)

                # Conditional loss: encourages generator to respect categorical constraints
                cross_entropy = cond_loss(faket, self.transformer.output_info, c, m)

                # Feature matching loss: align mean & std of discriminator features
                _,info_real = discriminator(real_cat_d)
                g = -torch.mean(y_fake) + cross_entropy
                g.backward(retain_graph=True)

                loss_mean = torch.norm(torch.mean(info_fake.view(self.batch_size,-1), dim=0) - torch.mean(info_real.view(self.batch_size,-1), dim=0), 1)
                loss_std = torch.norm(torch.std(info_fake.view(self.batch_size,-1), dim=0) - torch.std(info_real.view(self.batch_size,-1), dim=0), 1)
                loss_info = loss_mean + loss_std
                loss_info.backward()
                optimizerG.step()

                # (c) (Optional) Train Classifier if supervised
                if problem_type:
                    fake = self.generator(noisez)
                    faket = self.Gtransformer.inverse_transform(fake)
                    fakeact = apply_activate(faket, self.transformer.output_info)

                    real_pre, real_label = classifier(real)
                    fake_pre, fake_label = classifier(fakeact)

                    # Choose loss function depending on label type
                    c_loss = CrossEntropyLoss()
                    if (st_ed[1] - st_ed[0]) == 1: # regression
                        c_loss= SmoothL1Loss()
                        real_label = real_label.type_as(real_pre)
                        fake_label = fake_label.type_as(fake_pre)
                        real_label = torch.reshape(real_label,real_pre.size())
                        fake_label = torch.reshape(fake_label,fake_pre.size())

                    elif (st_ed[1] - st_ed[0]) == 2: # binary classification
                        c_loss = BCELoss()
                        real_label = real_label.type_as(real_pre)
                        fake_label = fake_label.type_as(fake_pre)

                    loss_cc = c_loss(real_pre, real_label)   # classifier loss on real
                    loss_cg = c_loss(fake_pre, fake_label)   # classifier loss on fake

                    # Update generator (improve fake label prediction)
                    optimizerG.zero_grad()
                    loss_cg.backward()
                    optimizerG.step()

                    # Update classifier (fit real labels)
                    optimizerC.zero_grad()
                    loss_cc.backward()
                    optimizerC.step()

            epoch += 1 # increment epoch counter
            # NOTE: uncomment following block if you want to calculate privacy budget (epsilon). Be careful, the calculation takes time, you don't want to
            # calculate it each epoch.

            # if self.private:
            #     max_lmbd = 4095
            #     lmbds = range(2, max_lmbd + 1)
            #     rdp = compute_rdp(self.micro_batch_size / train_data.shape[0], self.sigma, steps, lmbds)
            #     epsilon, _, _ = get_privacy_spent(lmbds, rdp, target_delta=1e-5)
            #     print("Epoch :", epoch, "Epsilon spent : ", epsilon)


    def sample(self, n):
        """
        Generate n synthetic samples using the trained generator.
        """
        # Put generator into evaluation mode (disables dropout, BN updates, etc.)
        self.generator.eval()

        # Get metadata about transformed columns (e.g., which are softmax/tanh)
        output_info = self.transformer.output_info

        # Number of generator passes needed to create at least n samples
        steps = n // self.batch_size + 1

        # List to collect generated batches
        data = []

        # First round of sampling
        for i in range(steps):
            # 1. Generate random noise (z)
            noisez = torch.randn(self.batch_size, self.random_dim, device=self.device)

            # 2. Sample conditional vectors (ensures categorical balance in sampling)
            condvec = self.cond_generator.sample(self.batch_size)
            c = condvec
            c = torch.from_numpy(c).to(self.device)

            # 3. Concatenate noise and conditional vector
            noisez = torch.cat([noisez, c], dim=1)
            # Reshape to (batch_size, latent_dim, 1, 1) so it matches ConvTranspose input
            noisez =  noisez.view(self.batch_size,self.random_dim+self.cond_generator.n_opt,1,1)

            # 4. Pass through generator → produce fake "image"
            fake = self.generator(noisez)

            # 5. Convert generator output from image back to flat vector
            faket = self.Gtransformer.inverse_transform(fake)

            # 6. Apply activation functions (tanh for continuous, gumbel-softmax for categorical)
            fakeact = apply_activate(faket,output_info)

            # 7. Move to CPU, numpy, and store
            data.append(fakeact.detach().cpu().numpy())

        # Concatenate all mini-batches into one big array
        data = np.concatenate(data, axis=0)

        # Convert GAN output back to real-world tabular values (inverse normalization + decode)
        result, resample = self.transformer.inverse_transform(data)

        # Resample loop (if not enough samples generated)
        while len(result) < n:
            data_resample = []

            # Number of extra steps needed
            steps_left = resample// self.batch_size + 1

            for i in range(steps_left):
                # Same process: noise → condvec → generator → inverse_transform
                noisez = torch.randn(self.batch_size, self.random_dim, device=self.device)
                condvec = self.cond_generator.sample(self.batch_size)
                c = condvec
                c = torch.from_numpy(c).to(self.device)
                noisez = torch.cat([noisez, c], dim=1)
                noisez =  noisez.view(self.batch_size,self.random_dim+self.cond_generator.n_opt,1,1)

                fake = self.generator(noisez)
                faket = self.Gtransformer.inverse_transform(fake)
                fakeact = apply_activate(faket, output_info)
                data_resample.append(fakeact.detach().cpu().numpy())

            # Merge resampled data
            data_resample = np.concatenate(data_resample, axis=0)

            # Decode back into real-world tabular form
            res,resample = self.transformer.inverse_transform(data_resample)

            # Append to final result
            result  = np.concatenate([result,res],axis=0)

        # Return exactly n rows (trim extra if over-generated)
        return result[0:n]


# ___________________________________________________
# Generative model training algorithm based on the CTABGANSynthesiser
class CTABGAN():
    def __init__(self, dataset_name, csv_dir):

        config = dataset_configs[dataset_name]

        self.__name__ = f'CTABGAN_{dataset_name}'
        self.raw_csv_path = f"{csv_dir}{dataset_name}.csv"
        self.raw_df = pd.read_csv(self.raw_csv_path)

        # Load config values dynamically
        self.categorical_columns = config.get("categorical_columns", [])
        self.log_columns = config.get("log_columns", [])
        self.mixed_columns = config.get("mixed_columns", {})
        self.general_columns = config.get("general_columns", [])
        self.non_categorical_columns = config.get("non_categorical_columns", [])
        self.integer_columns = config.get("integer_columns", [])
        self.problem_type = config.get("problem_type", {})

        self.test_ratio = 0.20
        self.synthesizer = CTABGANSynthesizer()

    def fit(self):
        # Preprocess the dataset and train the CTAB-GAN synthesizer.
        start_time = time.time() # start timer

        # Initialize preprocessing class to handle categorical encoding, normalization, etc.
        self.data_prep = DataPrep(
            self.raw_df,
            self.categorical_columns,
            self.log_columns,
            self.mixed_columns,
            self.general_columns,
            self.non_categorical_columns,
            self.integer_columns,
            self.problem_type,
            self.test_ratio
        )

        # Train the GAN synthesizer using the preprocessed dataset
        # Column types (categorical, mixed, general, etc.) are passed for proper handling
        self.synthesizer.fit(
            train_data=self.data_prep.df,
            categorical = self.data_prep.column_types["categorical"],
            mixed = self.data_prep.column_types["mixed"],
            general = self.data_prep.column_types["general"],
            non_categorical = self.data_prep.column_types["non_categorical"],
            type=self.problem_type
        )

        end_time = time.time() # end timer
        print('Finished training in',end_time-start_time," seconds.")


    def generate_samples(self):
        # Generate synthetic samples the same size as the original dataset
        # Sample synthetic data (same number of rows as real data)
        sample = self.synthesizer.sample(len(self.raw_df))

        # Convert GAN output back to human-readable format (inverse transforms)
        sample_df = self.data_prep.inverse_prep(sample)

        # Return synthetic dataset as a DataFrame
        return sample_df

# ___________________________________________________
def generate_ctabgan_synthetic(dataset_name, csv_dir, save_path):
    # Train CTABGAN and generate synthetic data for a single seed
    ctabgan = CTABGAN(dataset_name, csv_dir)  
    ctabgan.fit()
    
    synthetic_df = ctabgan.generate_samples()
    synthetic_df.to_csv(save_path, index=False)
    print(f"Synthetic data saved to {save_path}")

# ___________________________________________________
# Loop over datasets and seeds
for dataset_name in datasets: 
    for seed in seeds:
        synthetic_filename = f"CTABGAN_output_{dataset_name}_{seed}.csv"
        synthetic_path = os.path.join(filePath, synthetic_filename)
        generate_ctabgan_synthetic(dataset_name, basePath, synthetic_path)
