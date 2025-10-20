"""
Debug NAC column values
"""
import pandas as pd

# Load clinical data
clinical_df = pd.read_csv('patient_outcomes_mapping.csv')

# Find NAC column
for col in clinical_df.columns:
    if 'Neoadjuvant' in col and 'Received' in col:
        print(f"Column: {col}")
        print(f"Data type: {clinical_df[col].dtype}")
        print(f"Unique values: {clinical_df[col].unique()}")
        print(f"Value counts:")
        print(clinical_df[col].value_counts())
        print("\n")

# Also check chemotherapy columns
for col in clinical_df.columns:
    if 'Chemotherapy' in col:
        print(f"Column: {col}")
        print(f"Unique values: {clinical_df[col].unique()[:10]}")
        print(f"Value counts (top 5):")
        print(clinical_df[col].value_counts().head())
        print("\n")
