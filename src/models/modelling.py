import pandas as pd
import numpy as np
import pickle
import yaml
import logging
from typing import Tuple, Any
from sklearn.ensemble import RandomForestClassifier
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def load_params(params_path: str) -> dict:
    """
    Load parameters from a YAML file.
    Args:
        params_path (str): Path to the YAML file.
    Returns:
        dict: Parameters loaded from the YAML file.
    """
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.info(f"Parameters loaded from {params_path}")
        return params
    except Exception as e:
        logging.error(f"Failed to load parameters: {e}")
        raise

def load_train_data(train_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load training data from CSV file.
    Args:
        train_path (str): Path to the training CSV file.
    Returns:
        Tuple[np.ndarray, np.ndarray]: Features and labels.
    """
    try:
        train_data = pd.read_csv(train_path)
        x_train = train_data.drop(columns=['label']).values
        y_train = train_data['label'].values
        logging.info(f"Training data loaded from {train_path}")
        return x_train, y_train
    except Exception as e:
        logging.error(f"Failed to load training data: {e}")
        raise

def train_model(
    x_train: np.ndarray, 
    y_train: np.ndarray, 
    n_estimators: int, 
    max_depth: int
) -> RandomForestClassifier:
    """
    Train a RandomForestClassifier.
    Args:
        x_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        n_estimators (int): Number of trees.
        max_depth (int): Maximum tree depth.
    Returns:
        RandomForestClassifier: Trained model.
    """
    try:
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(x_train, y_train)
        logging.info("RandomForestClassifier trained successfully.")
        return model
    except Exception as e:
        logging.error(f"Model training failed: {e}")
        raise

def save_model(model: Any, model_path: str) -> None:
    """
    Save the trained model to a file.
    Args:
        model (Any): Trained model.
        model_path (str): Path to save the model.
    """
    try:
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logging.info(f"Model saved to {model_path}")
    except Exception as e:
        logging.error(f"Failed to save model: {e}")
        raise

def main() -> None:
    """
    Main function to orchestrate model training and saving.
    """
    try:
        params = load_params('params.yaml')
        n_estimators = params['modelling']['n_estimators']
        max_depth = params['modelling']['max_depth']
        x_train, y_train = load_train_data("data/interim/train_bow.csv")
        model = train_model(x_train, y_train, n_estimators, max_depth)
        save_model(model, "models/random_forest_model.pkl")
        logging.info("Model training and saving completed successfully.")
    except Exception as e:
        logging.error(f"Modelling pipeline failed: {e}")

if __name__ == "__main__":
    main()