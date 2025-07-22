import os
import logging
from typing import Tuple
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

def load_params(params_path: str) -> dict:
    """
    Load parameters from a YAML file.
    Args:
        params_path (str): Path to the YAML file.
    Returns:
        dict: Parameters loaded from the YAML file.
    Raises:
        FileNotFoundError: If the YAML file does not exist.
        yaml.YAMLError: If the YAML file is invalid.
    """
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.info(f"Parameters loaded from {params_path}")
        return params
    except FileNotFoundError as e:
        logging.error(f"Parameter file not found: {params_path}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {params_path}")
        raise

def load_dataset(url: str) -> pd.DataFrame:
    """
    Load dataset from a remote CSV file.
    Args:
        url (str): URL to the CSV file.
    Returns:
        pd.DataFrame: Loaded DataFrame.
    Raises:
        Exception: If loading fails.
    """
    try:
        df = pd.read_csv(url)
        logging.info(f"Dataset loaded from {url}")
        return df
    except Exception as e:
        logging.error(f"Failed to load dataset from {url}: {e}")
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter and map sentiment labels for binary classification.
    Args:
        df (pd.DataFrame): Original DataFrame.
    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    filtered_df = df[df['sentiment'].isin(['happiness', 'sadness'])].copy()
    filtered_df['sentiment'] = filtered_df['sentiment'].map({'happiness': 1, 'sadness': 0})
    logging.info("Data filtered and sentiment mapped to binary labels.")
    return filtered_df

def split_data(df: pd.DataFrame, test_size: float, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets.
    Args:
        df (pd.DataFrame): DataFrame to split.
        test_size (float): Proportion of test set.
        random_state (int): Random seed.
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Train and test DataFrames.
    """
    train_data, test_data = train_test_split(df, test_size=test_size, random_state=random_state)
    logging.info(f"Data split into train ({len(train_data)}) and test ({len(test_data)}) sets.")
    return train_data, test_data

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, output_dir: str) -> None:
    """
    Save train and test data to CSV files.
    Args:
        train_data (pd.DataFrame): Training data.
        test_data (pd.DataFrame): Testing data.
        output_dir (str): Directory to save CSV files.
    Raises:
        Exception: If saving fails.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        train_path = os.path.join(output_dir, 'train.csv')
        test_path = os.path.join(output_dir, 'test.csv')
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
        logging.info(f"Train and test data saved to {output_dir}")
    except Exception as e:
        logging.error(f"Failed to save data: {e}")
        raise

def main() -> None:
    """
    Main function to orchestrate data ingestion.
    """
    try:
        params = load_params('params.yaml')
        test_size = params['data_ingestion']['test_size']
        url = 'https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv'
        df = load_dataset(url)
        final_df = preprocess_data(df)
        train_data, test_data = split_data(final_df, test_size)
        save_data(train_data, test_data, 'data/raw')
        logging.info("Data ingestion completed successfully.")
    except Exception as e:
        logging.error(f"Data ingestion failed: {e}")

if __name__ == "__main__":
    main()