# This code dowloads the AI4I 2020 Predictive Maintenance dataset from the UCI ML Repository and saves it as a CSV file in the data folder.
from ucimlrepo import fetch_ucirepo
import pandas as pd
import os

# Create data folder if it doesn't exist
os.makedirs("data", exist_ok=True)

# Fetch the dataset from UCI ML Repository using the ucimlrepo package. The dataset is identified by its ID (601).
ds = fetch_ucirepo(id=601)

# Combine the features and targets into a single DataFrame, clean the column names, and save it as a CSV file.
df = pd.concat([ds.data.features, ds.data.targets], axis=1)
df.columns = [col.strip().lower().replace(" ", "_").replace("[", "").replace("]", "") for col in df.columns]
df.to_csv("data/ai4i2020.csv", index=False)


print(f"Dataset with: {len(df)} rows and {len(df.columns)} columns")
print(f"Failures: {df['machine_failure'].sum()} ({df['machine_failure'].mean() * 100:.1f}%)")
print(df.describe())