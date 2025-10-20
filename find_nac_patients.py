"""
Find patients who received neoadjuvant chemotherapy (NAC)
"""
import pandas as pd
import os
from pathlib import Path

# Load clinical data
clinical_df = pd.read_csv('patient_outcomes_mapping.csv')

# Find patient ID column
patient_id_col = None
for col in clinical_df.columns:
    if 'Patient' in col and 'ID' in col:
        patient_id_col = col
        break

# Find NAC column
nac_col = None
for col in clinical_df.columns:
    if 'Neoadjuvant therapy - Received Neoadjuvant Therapy or Not' in col:
        nac_col = col
        break

if patient_id_col and nac_col:
    # Filter patients who received NAC (value = 1)
    nac_patients = clinical_df[clinical_df[nac_col] == 1]
    
    print(f"Total patients in dataset: {len(clinical_df)}")
    print(f"Patients who received NAC: {len(nac_patients)}")
    print(f"\nFirst 20 NAC patients:")
    print(nac_patients[patient_id_col].head(20).tolist())
    
    # Check which of these have DICOM folders
    dicom_root = Path("C:/Users/kanthamnenm/Downloads/Segmentation_Input_Automated_Download")
    available_folders = [f.name for f in dicom_root.iterdir() if f.is_dir()]
    
    # Extract patient IDs from folder names
    nac_patient_ids = nac_patients[patient_id_col].tolist()
    available_nac = []
    
    for folder in available_folders:
        # Extract patient ID (e.g., Breast_MRI_001 from folder name)
        parts = folder.split('_')
        if len(parts) >= 3:
            patient_id = f"{parts[0]}_{parts[1]}_{parts[2]}"
            if patient_id in nac_patient_ids:
                available_nac.append((patient_id, folder))
    
    print(f"\nNAC patients with DICOM data available: {len(available_nac)}")
    print(f"\nFirst 10 available NAC patients:")
    for patient_id, folder in available_nac[:10]:
        print(f"  {patient_id}: {folder}")
    
    # Save list of NAC patients
    nac_df = nac_patients[[patient_id_col]]
    nac_df.to_csv('nac_patients_list.csv', index=False)
    print(f"\n✓ Saved list of NAC patients to nac_patients_list.csv")
else:
    print("Could not find required columns")
    print(f"Columns: {list(clinical_df.columns)}")
