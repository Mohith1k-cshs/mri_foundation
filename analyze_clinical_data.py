"""
Analyze clinical data structure for TNM staging prediction
"""
import pandas as pd
import numpy as np

# Load the clinical data
print("Loading clinical data...")
clinical_df = pd.read_excel('Clinical_and_Other_Features.xlsx')

print("=== CLINICAL DATA OVERVIEW ===")
print(f"Total patients: {len(clinical_df)}")
print(f"\nColumn names ({len(clinical_df.columns)} total):")
for i, col in enumerate(clinical_df.columns):
    print(f"  {i+1}. {col}")

print(f"\n=== FIRST 5 ROWS ===")
print(clinical_df.head())

print(f"\n=== DATA TYPES ===")
print(clinical_df.dtypes)

print(f"\n=== MISSING VALUES ===")
missing = clinical_df.isnull().sum()
print(missing[missing > 0])

print(f"\n=== TNM STAGING COLUMNS (if present) ===")
tnm_keywords = ['pT', 'pt', 'pN', 'pn', 'pM', 'pm', 'stage', 'Stage', 'tumor', 'Tumor', 
                'node', 'Node', 'metastasis', 'Metastasis', 'pCR', 'pcr', 'RCB', 'rcb',
                'response', 'Response', 'chemotherapy', 'Chemotherapy', 'NAC', 'nac']

tnm_cols = []
for col in clinical_df.columns:
    col_str = str(col)
    if any(keyword in col_str for keyword in tnm_keywords):
        tnm_cols.append(col)
        print(f"\n  Found: {col}")
        print(f"    Unique values: {clinical_df[col].nunique()}")
        print(f"    Value counts:")
        print(f"    {clinical_df[col].value_counts().head(10)}")

print(f"\n=== PATIENT ID COLUMNS ===")
id_keywords = ['ID', 'id', 'Id', 'Patient', 'patient', 'Subject', 'subject', 'MRI', 'mri']
for col in clinical_df.columns:
    col_str = str(col)
    if any(keyword in col_str for keyword in id_keywords):
        print(f"  {col}: {clinical_df[col].head(5).tolist()}")

print(f"\n=== SUMMARY STATISTICS ===")
print(clinical_df.describe())

# Save a summary report
with open('clinical_data_summary.txt', 'w') as f:
    f.write(f"Total patients: {len(clinical_df)}\n\n")
    f.write("Columns:\n")
    for col in clinical_df.columns:
        f.write(f"  - {col}\n")
    f.write(f"\nTNM-related columns found: {len(tnm_cols)}\n")
    for col in tnm_cols:
        f.write(f"  - {col}\n")

print("\n✓ Analysis complete! Summary saved to clinical_data_summary.txt")
