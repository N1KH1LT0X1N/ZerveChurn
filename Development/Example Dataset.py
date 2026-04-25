import pandas as pd

# Load the dataset from the file system
user_retention = pd.read_parquet('user_retention.parquet')

# Preview the dataset
print(f"Dataset shape: {user_retention.shape}")
print(f"\nFirst few rows:")
user_retention