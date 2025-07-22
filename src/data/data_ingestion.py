import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split

# Load the dataset from a remote CSV file
df = pd.read_csv('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')

# Filter the DataFrame to only include rows with 'happiness' or 'sadness' sentiments
final_df = df[df['sentiment'].isin(['happiness', 'sadness'])].copy()

# Map 'happiness' to 1 and 'sadness' to 0 for binary classification
final_df['sentiment'] = final_df['sentiment'].map({'happiness': 1, 'sadness': 0})

# Split the data into training and testing sets (80% train, 20% test)
train_data, test_data = train_test_split(final_df, test_size=0.2, random_state=42)

# Create the directory to save raw data if it doesn't exist
os.makedirs('data/raw', exist_ok=True)

# Save the training and testing data to CSV files
train_data.to_csv('data/raw/train.csv', index=False)
test_data.to_csv('data/raw/test.csv', index=False)