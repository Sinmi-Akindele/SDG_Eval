# Import methods
import os
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
warnings.filterwarnings('ignore')

# Variable declaration
seeds = [42, 50, 61, 79, 83]
ratios = [0.8, 0.6, 0.4, 0.2]

# Paths for file
basePath = # INSERT PATH
filePath = # INSERT PATH
synthPath = # INSERT PATH

# Define synthetic generators and datasets
models_and_datasets = {
    'CTABGAN': ["ACSHI", "adult", "australian", "compas-scores-two-years", "german_data"],
    'TabDDPM': ["ACSHI", "adult", "australian", "compas-scores-two-years", "german_data"],
    'TabSyn': ["ACSHI", "adult", "australian", "compas-scores-two-years", "german_data"]
}

# Define dataset attributes
dataset_columns = {
    "ACSHI": {
        "continuous": ['AGEP', 'SCHL', 'ESP', 'CIT', 'MIG', 'MIL',
                      'ANC', 'DREM', 'PINCP', 'ESR'],
        "categorical": ['SEX', 'DEAR', 'DEYE', 'FER'],
        "minority_column": 'SEX',
        "minority_value": 1,
        "target_column": 'label'
    },
    "adult": {
        "continuous": ['age', 'education-num', 'capital-gain', 'capital-loss',
                       'hours-per-week'],
        "categorical": ['education', 'workclass', 'marital-status', 'occupation',
                        'relationship', 'race', 'sex', 'native-country'],
        "minority_column": 'sex',
        "minority_value": 'Female',
        "target_column": 'income-per-year'
    },
    "australian": {
        "continuous": ['A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A10', 'A12', 'A13', 'A14'],
        "categorical": ['A1', 'A8', 'A9', 'A11'],
        "minority_column": 'A15',
        "minority_value": 1,
        "target_column": 'A15'
    },
    "compas-scores-two-years": {
        "continuous": ['age',  'priors_count', 'juv_other_count'],
        "categorical": ['sex', 'age_cat', 'race', 'juv_fel_count', 'juv_misd_count',
                        'c_charge_degree'],
        "minority_column": 'sex',
        "minority_value": 'Female',
        "target_column": 'two_year_recid'
    },
    "german_data": {
        "continuous": ['month', 'credit_amount', 'investment_as_income_percentage',
                       'residence_since', 'age', 'number_of_credits', 'people_liable_for'],
        "categorical": ['status', 'credit_history', 'purpose', 'savings', 'employment', 'sex',
                        'other_debtors', 'property', 'installment_plans', 'housing',
                        'skill_level', 'telephone', 'foreign_worker'],
        "minority_column": 'sex',
        "minority_value": 'female',
        "target_column": 'credit'
    }
}

# Define Logistic Regression model class
class LogisticRegression:
    def __init__(self, scoring):
        self.scoring = scoring
        self.name = "logistic_regression"

    def split(self, data):
        # Enforce 80:20 train-test split
        return train_test_split(data, test_size=0.2, random_state=42)

    def fit(self, dataset_name, train_data):
        # Train Logistic Regression model with hyperparameter tuning
        dataset_info = dataset_columns[dataset_name]
        categorical_columns = dataset_info["categorical"]
        numerical_columns = dataset_info["continuous"]
        target_column = dataset_info["target_column"]

        y_train = train_data[target_column]
        X_train = train_data.drop(columns=[target_column])

        feature_transformation = ColumnTransformer(transformers=[
            ('categorical_features', OneHotEncoder(handle_unknown='ignore'), categorical_columns),
            ('normalized_numeric', MinMaxScaler(), numerical_columns),
        ], sparse_threshold=0.3)

        param_grid = {
            'learner__loss': ['log_loss'],
            'learner__penalty': ['l2', 'l1', 'elasticnet'],
            'learner__alpha': [0.0001, 0.001, 0.01, 0.1]
        }

        pipeline = Pipeline([
            ('features', feature_transformation),
            ('learner', SGDClassifier(max_iter=1000))
        ])

        search = GridSearchCV(pipeline, param_grid, scoring=self.scoring, cv=5, verbose=1, n_jobs=-1)
        model = search.fit(X_train, y_train)

        return model

# Define XGBoost model class
class XGBoostModel:
    def __init__(self, scoring):
        self.scoring = scoring
        self.name = "xgboost"

    def split(self, data):
        # Enforce 80:20 train-test split
        return train_test_split(data, test_size=0.2, random_state=42)

    def fit(self, dataset_name, train_data):
        # Train XGBoost model with hyperparameter tuning
        dataset_info = dataset_columns[dataset_name]
        categorical_columns = dataset_info["categorical"]
        numerical_columns = dataset_info["continuous"]
        target_column = dataset_info["target_column"]

        y_train = train_data[target_column]
        X_train = train_data.drop(columns=[target_column])

        # Preprocessing pipeline
        feature_transformation = ColumnTransformer(
            transformers=[
                ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical_columns),
                ('numerical', MinMaxScaler(), numerical_columns),
            ],
            sparse_threshold=0.3
        )

        param_grid = {
            'learner__n_estimators': [5, 10],
            'learner__max_depth': [3, 6, 10],
            'learner__objective': ['binary:logistic']
        }

        pipeline = Pipeline([
            ('features', feature_transformation),
            ('learner', xgb.XGBClassifier())
        ])

        search = GridSearchCV(pipeline, param_grid, scoring=self.scoring, cv=5, verbose=1, n_jobs=-1)
        model = search.fit(X_train, y_train)

        return model

# Define MLP model class
class MLPModel:
    def __init__(self, scoring):
        self.scoring = scoring
        self.name = "mlp"

    def split(self, data):
        # Perform 80:20 train-test split
        return train_test_split(data, test_size=0.2, random_state=42)

    def fit(self, dataset_name, train_data):
        # Train MLPClassifier model with hyperparameter tuning
        dataset_info = dataset_columns[dataset_name]
        categorical_columns = dataset_info["categorical"]
        numerical_columns = dataset_info["continuous"]
        target_column = dataset_info["target_column"]

        y_train = train_data[target_column]
        X_train = train_data.drop(columns=[target_column])

        # Preprocessing pipeline
        feature_transformation = ColumnTransformer(
            transformers=[
                ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical_columns),
                ('numerical', MinMaxScaler(), numerical_columns),
            ],
            sparse_threshold=0.3
        )

        param_grid = {
            'learner__hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'learner__activation': ['relu', 'tanh'],
            'learner__solver': ['adam', 'sgd'],
            'learner__max_iter': [200, 500]
        }

        pipeline = Pipeline([
            ('features', feature_transformation),
            ('learner', MLPClassifier(random_state=42))
        ])

        search = GridSearchCV(pipeline, param_grid, scoring=self.scoring, cv=5, verbose=1, n_jobs=-1)
        model = search.fit(X_train, y_train)

        return model

# Define ml models and score
models_and_scores = [
    LogisticRegression(scoring="accuracy"),
    XGBoostModel(scoring="accuracy"),
    MLPModel(scoring="accuracy")
]

def run_experiment_for_all_datasets(datasets_info, base_path, filePath, synthPath, models_and_datasets, seeds, ratios, scoring):
    results = {}

    # Iterate through each dataset
    for dataset_name, dataset_info in datasets_info.items():
        print(f"\nRunning experiments for {dataset_name}...")

        # Load dataset 
        data_path = os.path.join(base_path, f"{dataset_name}.csv")
        df = pd.read_csv(data_path)

        dataset_results = {}
        # Train and evaluate model on real data
        for model in models_and_scores:
            model_name = model.name
            print(f"Training {model_name}...")

            # Split the data into training and testing sets
            train_data, test_data = model.split(df)

            # Train the model
            trained_model = model.fit(dataset_name, train_data)

            # Get predictions for the test set
            X_test = test_data.drop(columns=[dataset_info['target_column']])
            y_test = test_data[dataset_info['target_column']]
            y_pred = trained_model.predict(X_test)

            # Extract minority column values for test set
            minority_test_real = test_data[dataset_info['minority_column']].values

            # Store real data predictions
            real_predictions_df = pd.DataFrame({
                dataset_info['minority_column']: minority_test_real,
                'y_true': y_test.values,
                'y_pred': y_pred
            })

            # Save real predictions
            real_predictions_file_path = os.path.join(filePath, f"{dataset_name}_{model_name}_real_predictions.csv")
            real_predictions_df.to_csv(real_predictions_file_path, index=False)

            dataset_results[f"{model_name}_real"] = real_predictions_df

            # Evaluate the trained model on synthetic datasets
            for synthetic_model, datasets in models_and_datasets.items():
                if dataset_name in datasets:
                    for seed in seeds:
                        # Load synthetic dataset
                        synthetic_data_path = os.path.join(synthPath, f"{synthetic_model}_output_{dataset_name}_{seed}.csv")
                        synthetic_df = pd.read_csv(synthetic_data_path)

                        # Ensure columns match between real and synthetic datasets
                        X_synthetic = synthetic_df.drop(columns=[dataset_info['target_column']])
                        y_synthetic_true = synthetic_df[dataset_info['target_column']]

                        # Predict using the trained model
                        y_synthetic_pred = trained_model.predict(X_synthetic)

                        # Extract minority column values for synthetic test set
                        minority_test_synthetic = synthetic_df[dataset_info['minority_column']].values
  
                        # Store synthetic data predictions
                        synthetic_predictions_df = pd.DataFrame({
                            dataset_info['minority_column']: minority_test_synthetic,
                            'y_true': y_synthetic_true.values,
                            'y_pred': y_synthetic_pred
                        })

                        # Save synthetic predictions
                        synthetic_predictions_file_path = os.path.join(filePath, f"{synthetic_model}_output_{dataset_name}_{model_name}_predictions_seed_{seed}.csv")
                        synthetic_predictions_df.to_csv(synthetic_predictions_file_path, index=False)

                        # Save results
                        dataset_results[f"{model_name}_{dataset_name}_{synthetic_model}_seed_{seed}"] = synthetic_predictions_df

        results[dataset_name] = dataset_results

    return results

# Run the experiment for all datasets
results = run_experiment_for_all_datasets(
    dataset_columns,
    basePath,
    filePath,
    synthPath,
    models_and_datasets,
    seeds,
    ratios,
    scoring="accuracy"
)
