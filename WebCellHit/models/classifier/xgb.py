import pandas as pd
from sklearn.model_selection import train_test_split

import xgboost as xgb
from xgboost import XGBClassifier
import optuna
import numpy as np

import argparse

class AutoXGBClassifier:
    
    def __init__(self, num_parallel_tree=5, gpuID=0, objective_type='multi:softmax',num_classes=None):
        self.best_params = None
        self.model = XGBClassifier()
        self.best_trial = None
        self.study = None
        self.num_parallel_tree = num_parallel_tree
        self.gpuID = gpuID
        self.objective_type = objective_type
        self.num_classes = num_classes

    def objective(self, trial, cv_data=None):
        """
        Objective function for Optuna study.
        Optimizes hyperparameters to minimize the evaluation metric.
        """
        # Define the hyperparameter search space
        params = {
            'eta': trial.suggest_float('eta', 0.01, 1.0),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, log=True),
            'max_depth': trial.suggest_int('max_depth', 1, 20),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 1.0),
            'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 1, 10),
            'lambda': trial.suggest_float('lambda', 0.5, 3.0),
            'sampling_method': trial.suggest_categorical('sampling_method', ['uniform', 'gradient_based']),
            'num_parallel_tree': self.num_parallel_tree,
            'tree_method': 'hist',  # Enable GPU acceleration for classification
            'device': f'cuda:{self.gpuID}',       # Specify GPU ID
            'objective': self.objective_type,
            'eval_metric': 'mlogloss',    # Evaluation metric for binary classification
            'num_class': self.num_classes
        }

        # List to store validation scores for each fold
        validation_scores = []

        # Iterate over each cross-validation fold
        for (train_X, train_Y), (valid_X, valid_Y) in cv_data:
            # Create DMatrix for training and validation
            dtrain = xgb.DMatrix(train_X, label=train_Y['project_id'])
            dval = xgb.DMatrix(valid_X, label=valid_Y['project_id'])
            
            # Exclude parameters that are not used in the training function
            train_params = {k: v for k, v in params.items() if k not in ['n_estimators', 'early_stopping_rounds']}
            
            # Train the model with early stopping
            booster = xgb.train(
                params=train_params,
                dtrain=dtrain,
                num_boost_round=params['n_estimators'],
                evals=[(dval, 'eval')],
                early_stopping_rounds=params['early_stopping_rounds'],
                verbose_eval=False
            )
            
            # Retrieve the best score for the evaluation metric
            best_score = booster.best_score
            validation_scores.append(best_score)

        # Return the mean validation score across all folds
        return np.mean(validation_scores)

    def search(self, 
               cv_data=None, 
               n_trials=300, 
               n_startup_trials=100,
               optim_seed=0,
               storage=None,
               study_name=None):
        """
        Perform hyperparameter search using Optuna.
        
        Parameters:
        - cv_data: Iterable containing cross-validation splits as ((train_X, train_Y), (valid_X, valid_Y))
        - n_trials: Total number of trials for the optimization
        - n_startup_trials: Number of trials for the sampler to run before optimizing
        - optim_seed: Seed for the sampler for reproducibility
        """
        
        # Initialize the Optuna sampler
        sampler = optuna.samplers.TPESampler(
            seed=optim_seed, 
            n_startup_trials=n_startup_trials,
            multivariate=True
        )
        
        # Create the Optuna study
        self.study = optuna.create_study(
            direction="minimize",  # Since we're minimizing logloss
            sampler=sampler,
            storage=storage,
            study_name=study_name,
            load_if_exists=True
            # You can add storage and study_name if needed
        )

        # Optimize the study using the objective function
        self.study.optimize(
            lambda trial: self.objective(trial, cv_data=cv_data), 
            n_trials=n_trials
        )
        

    def get_best_params(self):
        """
        Retrieve the best hyperparameters found during the search.
        """
        if self.study:
            self.best_params = self.study.best_trial.params
            return self.best_params
        else:
            raise ValueError("Study has not been conducted yet. Call the search method first.")

    def train_final_model(self, X, Y, X_val=None, Y_val=None):
        """
        Train the final XGBClassifier model using the best hyperparameters.
        
        Parameters:
        - X: Training features
        - Y: Training labels
        - X_val: Validation features (optional)
        - Y_val: Validation labels (optional)
        """
        if self.best_params is None:
            self.get_best_params()

        constructor_params = {k: v for k, v in self.best_params.items() if k not in ['n_estimators', 'early_stopping_rounds']}
        constructor_params['objective'] = self.objective_type
        constructor_params['eval_metric'] = 'mlogloss'    # Evaluation metric for binary classification
        constructor_params['num_class'] = self.num_classes
        constructor_params['device'] = f'cuda:{self.gpuID}'

        #make dmatrices
        dtrain = xgb.DMatrix(X, label=Y)
        dval = xgb.DMatrix(X_val, label=Y_val)

        # Prepare evaluation set if validation data is provided
        if X_val is not None and Y_val is not None:
            eval_set = [(dval, 'eval')]
        else:
            eval_set = None
        
        # Update the model with the best hyperparameters
        self.model = xgb.train(
            params=constructor_params,
            dtrain=dtrain,
            num_boost_round=self.best_params['n_estimators'],
            evals=eval_set,
            early_stopping_rounds=self.best_params['early_stopping_rounds'],
            verbose_eval=False
        )

    def save_model(self, filepath):
        """
        Save the trained XGBClassifier model to a file.
        
        Parameters:
        - filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call the train_final_model method first.")
        
        self.model.save_model(filepath)
        print(f"Model saved successfully to {filepath}")

    def load_model(self, filepath):
        """
        Load a trained XGBClassifier model from a file.
        
        Parameters:
        - filepath: Path to the saved model
        """
        self.model = XGBClassifier()
        self.model.load_model(filepath)
        print(f"Model loaded successfully from {filepath}")

    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Parameters:
        - X: Features to predict
        
        Returns:
        - Predictions
        """
        if not self.ensemble:
            raise ValueError("Model has not been trained yet. Call the train_final_model method first.")
        
        return self.model.predict(X)