"""
Properly parse clinical data with multi-row headers
"""
import pandas as pd
import numpy as np

# Load the clinical data - skip the first row which seems to be category headers
print("Loading clinical data...")
clinical_df = pd.read_excel('Clinical_and_Other_Features.xlsx', header=[0, 1])

print("=== CLINICAL DATA OVERVIEW ===")
print(f"Total patients: {len(clinical_df)}")

# Flatten multi-level columns
clinical_df.columns = [' - '.join([str(c) for c in col if 'Unnamed' not in str(c)]).strip(' - ') 
                        for col in clinical_df.columns.values]

print(f"\nColumn names ({len(clinical_df.columns)} total):")
for i, col in enumerate(clinical_df.columns):
    if i < 50:  # Show first 50
        print(f"  {i+1}. {col}")

# Find Patient ID column
patient_id_col = None
for col in clinical_df.columns:
    if 'Patient' in col and 'ID' in col:
        patient_id_col = col
        break

if patient_id_col:
    print(f"\n=== PATIENT IDs (first 10) ===")
    print(clinical_df[patient_id_col].head(10))

# Look for TNM staging and response columns
print(f"\n=== SEARCHING FOR TNM AND RESPONSE COLUMNS ===")
keywords = {
    'pT': ['pT', 'pathologic stage', 'post-neoadjuvant T', 'T stage', 'T-stage'],
    'pN': ['pN', 'nodal', 'lymph node', 'N stage', 'N-stage'],
    'pM': ['pM', 'metastasis', 'M stage', 'M-stage', 'distant'],
    'pCR': ['pCR', 'complete response', 'pathologic response'],
    'RCB': ['RCB', 'residual cancer burden'],
    'NAC': ['neoadjuvant', 'NAC', 'chemotherapy'],
    'Response': ['response', 'Response']
}

found_cols = {}
for category, search_terms in keywords.items():
    matches = []
    for col in clinical_df.columns:
        col_lower = col.lower()
        if any(term.lower() in col_lower for term in search_terms):
            matches.append(col)
    if matches:
        found_cols[category] = matches
        print(f"\n{category} related columns:")
        for match in matches:
            print(f"  - {match}")
            print(f"    Unique values: {clinical_df[match].nunique()}")
            if clinical_df[match].nunique() < 20:
                print(f"    Values: {clinical_df[match].value_counts().head()}")

# Save cleaned data
print("\n=== SAVING PROCESSED DATA ===")
clinical_df.to_csv('clinical_data_processed.csv', index=False)
print("✓ Saved to clinical_data_processed.csv")

# Create a mapping file for patient IDs to outcomes
if patient_id_col and found_cols:
    outcome_cols = [patient_id_col]
    for category, cols in found_cols.items():
        outcome_cols.extend(cols)
    
    outcomes_df = clinical_df[outcome_cols].copy()
    outcomes_df.to_csv('patient_outcomes_mapping.csv', index=False)
    print("✓ Saved patient outcomes to patient_outcomes_mapping.csv")

print("\n=== ANALYSIS COMPLETE ===")
