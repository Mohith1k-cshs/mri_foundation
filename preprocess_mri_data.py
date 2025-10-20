"""
Data Preprocessing Pipeline for MRI-CORE Clinical Prediction

This script processes:
1. DICOM files from Segmentation_Input_Automated_Download
2. NRRD segmentation masks from Segmentation_Outputs_293
3. Clinical outcomes from patient_outcomes_mapping.csv

Output: Preprocessed 2D slices with labels ready for training
"""

import os
import numpy as np
import pydicom
import nrrd
import pandas as pd
from pathlib import Path
import SimpleITK as sitk
from PIL import Image
import json
from tqdm import tqdm

class MRIPreprocessor:
    def __init__(self,
                 dicom_root="C:/Users/kanthamnenm/Downloads/Segmentation_Input_Automated_Download",
                 nrrd_root="C:/Users/kanthamnenm/Downloads/Segmentation_Outputs_293",
                 clinical_csv="patient_outcomes_mapping.csv",
                 output_dir="preprocessed_data"):
        
        self.dicom_root = Path(dicom_root)
        self.nrrd_root = Path(nrrd_root)
        self.clinical_csv = clinical_csv
        self.output_dir = Path(output_dir)
        
        # Create output directories
        self.images_dir = self.output_dir / "images"
        self.masks_dir = self.output_dir / "masks"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        
        # Load clinical data
        self.load_clinical_data()
        
    def load_clinical_data(self):
        """Load and process clinical outcomes"""
        print("Loading clinical data...")
        self.clinical_df = pd.read_csv(self.clinical_csv)
        
        # Find relevant columns
        self.patient_id_col = None
        for col in self.clinical_df.columns:
            if 'Patient' in col and 'ID' in col:
                self.patient_id_col = col
                break
        
        if self.patient_id_col is None:
            raise ValueError("Could not find Patient ID column in clinical data")
        
        print(f"Found {len(self.clinical_df)} patients in clinical data")
        print(f"Patient ID column: {self.patient_id_col}")
        
    def extract_patient_id(self, folder_name):
        """Extract patient ID from folder name (e.g., 'Breast_MRI_001...' -> 'Breast_MRI_001')"""
        # Format: Breast_MRI_XXX_...
        parts = folder_name.split('_')
        if len(parts) >= 3:
            return f"{parts[0]}_{parts[1]}_{parts[2]}"
        return None
    
    def load_dicom_series(self, patient_folder):
        """Load DICOM series from patient folder"""
        try:
            # Read the DICOM series using SimpleITK
            reader = sitk.ImageSeriesReader()
            dicom_files = reader.GetGDCMSeriesFileNames(str(patient_folder))
            
            if len(dicom_files) == 0:
                print(f"  Warning: No DICOM files found in {patient_folder}")
                return None
            
            reader.SetFileNames(dicom_files)
            image = reader.Execute()
            
            # Convert to numpy array (Z, Y, X)
            volume = sitk.GetArrayFromImage(image)
            
            return volume
        
        except Exception as e:
            print(f"  Error loading DICOM from {patient_folder}: {e}")
            return None
    
    def load_nrrd_mask(self, patient_id):
        """Load NRRD segmentation mask"""
        # Format: Segmentation_Breast_MRI_XXX_Breast.seg.nrrd
        mask_file = self.nrrd_root / f"Segmentation_{patient_id}_Breast.seg.nrrd"
        
        if not mask_file.exists():
            return None
        
        try:
            data, header = nrrd.read(str(mask_file))
            return data
        except Exception as e:
            print(f"  Error loading NRRD mask {mask_file}: {e}")
            return None
    
    def normalize_slice(self, slice_2d):
        """Normalize slice to [0, 1] range"""
        slice_min = slice_2d.min()
        slice_max = slice_2d.max()
        
        if slice_max - slice_min == 0:
            return np.zeros_like(slice_2d, dtype=np.float32)
        
        normalized = (slice_2d - slice_min) / (slice_max - slice_min)
        return normalized.astype(np.float32)
    
    def save_slice_as_png(self, slice_2d, output_path):
        """Save normalized slice as PNG"""
        # Convert to 8-bit image
        img_8bit = (slice_2d * 255).astype(np.uint8)
        img = Image.fromarray(img_8bit).convert('RGB')  # Convert to RGB for consistency
        img.save(output_path)
    
    def get_clinical_labels(self, patient_id):
        """Get clinical outcome labels for a patient"""
        patient_data = self.clinical_df[self.clinical_df[self.patient_id_col] == patient_id]
        
        if len(patient_data) == 0:
            return None
        
        # Extract TNM staging and response labels
        labels = {}
        
        # Map column names (simplified - adjust based on actual column names)
        label_mapping = {
            'pT': 'Pathologic Response to Neoadjuvant Therapy - Pathologic response to Neoadjuvant therapy: Pathologic stage (T) following neoadjuvant therapy',
            'pN': 'Pathologic Response to Neoadjuvant Therapy - Pathologic response to Neoadjuvant therapy:  Pathologic stage (N) following neoadjuvant therapy',
            'pM': 'Pathologic Response to Neoadjuvant Therapy - Pathologic response to Neoadjuvant therapy:  Pathologic stage (M) following neoadjuvant therapy',
            'pCR': 'Tumor Response - Pathologic Response to Neoadjuvant Therapy',
            'near_complete': 'Near Complete Response - Overall Near-complete Response:  Stricter Definition',
            'received_nac': 'Neoadjuvant therapy - Received Neoadjuvant Therapy or Not'
        }
        
        for key, col_name in label_mapping.items():
            if col_name in patient_data.columns:
                value = patient_data[col_name].values[0]
                labels[key] = value
        
        return labels
    
    def process_patient(self, patient_folder):
        """Process one patient's data"""
        patient_folder_name = patient_folder.name
        patient_id = self.extract_patient_id(patient_folder_name)
        
        if patient_id is None:
            print(f"  Could not extract patient ID from {patient_folder_name}")
            return None
        
        print(f"Processing {patient_id}...")
        
        # Get clinical labels
        labels = self.get_clinical_labels(patient_id)
        if labels is None:
            print(f"  No clinical data found for {patient_id}")
            return None
        
        # Check if patient received NAC
        if 'received_nac' in labels:
            # Convert to string for comparison (values are stored as strings '1' or '2')
            if str(labels['received_nac']) != '1':  # '1' = yes, '2' = no
                print(f"  Patient {patient_id} did not receive NAC, skipping...")
                return None
        
        # Load DICOM volume
        volume = self.load_dicom_series(patient_folder)
        if volume is None:
            return None
        
        # Load segmentation mask (optional)
        mask = self.load_nrrd_mask(patient_id)
        if mask is not None:
            print(f"  Loaded segmentation mask: {mask.shape}")
        
        # Process each slice
        slice_data = []
        for slice_idx in range(volume.shape[0]):
            slice_2d = volume[slice_idx, :, :]
            
            # Skip empty slices
            if np.max(slice_2d) == 0:
                continue
            
            # Normalize
            normalized_slice = self.normalize_slice(slice_2d)
            
            # Save as PNG
            slice_filename = f"{patient_id}_slice_{slice_idx:03d}.png"
            slice_path = self.images_dir / slice_filename
            self.save_slice_as_png(normalized_slice, slice_path)
            
            # Save mask if available
            mask_filename = None
            if mask is not None and slice_idx < mask.shape[0]:
                mask_slice = mask[slice_idx, :, :]
                mask_filename = f"{patient_id}_slice_{slice_idx:03d}_mask.png"
                mask_path = self.masks_dir / mask_filename
                # Save mask as binary image
                mask_img = Image.fromarray((mask_slice > 0).astype(np.uint8) * 255)
                mask_img.save(mask_path)
            
            slice_data.append({
                'patient_id': patient_id,
                'slice_idx': slice_idx,
                'image_path': str(slice_path.relative_to(self.output_dir)),
                'mask_path': str(mask_path.relative_to(self.output_dir)) if mask_filename else None,
                **labels
            })
        
        print(f"  Processed {len(slice_data)} slices")
        return slice_data
    
    def run(self, max_patients=None):
        """Run preprocessing pipeline"""
        print("="*80)
        print("MRI-CORE Clinical Prediction - Data Preprocessing")
        print("="*80)
        
        # Get all patient folders
        patient_folders = sorted([f for f in self.dicom_root.iterdir() if f.is_dir()])
        
        if max_patients is not None:
            patient_folders = patient_folders[:max_patients]
        
        print(f"\nFound {len(patient_folders)} patient folders")
        print(f"Output directory: {self.output_dir}")
        
        # Process each patient
        all_slice_data = []
        successful = 0
        failed = 0
        
        for patient_folder in tqdm(patient_folders, desc="Processing patients"):
            try:
                slice_data = self.process_patient(patient_folder)
                if slice_data:
                    all_slice_data.extend(slice_data)
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"  ERROR processing {patient_folder.name}: {e}")
                failed += 1
        
        # Save metadata
        metadata_df = pd.DataFrame(all_slice_data)
        metadata_path = self.output_dir / "metadata.csv"
        metadata_df.to_csv(metadata_path, index=False)
        
        # Save summary
        summary = {
            'total_patients': len(patient_folders),
            'successful': successful,
            'failed': failed,
            'total_slices': len(all_slice_data),
            'output_dir': str(self.output_dir),
            'label_distributions': {
                'pT': metadata_df['pT'].value_counts().to_dict() if 'pT' in metadata_df.columns else {},
                'pN': metadata_df['pN'].value_counts().to_dict() if 'pN' in metadata_df.columns else {},
                'pM': metadata_df['pM'].value_counts().to_dict() if 'pM' in metadata_df.columns else {},
                'pCR': metadata_df['pCR'].value_counts().to_dict() if 'pCR' in metadata_df.columns else {},
            }
        }
        
        summary_path = self.output_dir / "preprocessing_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "="*80)
        print("PREPROCESSING COMPLETE")
        print("="*80)
        print(f"Successful: {successful}/{len(patient_folders)} patients")
        print(f"Total slices: {len(all_slice_data)}")
        print(f"Metadata saved to: {metadata_path}")
        print(f"Summary saved to: {summary_path}")
        
        return metadata_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess MRI data for clinical prediction")
    parser.add_argument('--max_patients', type=int, default=None, 
                        help='Maximum number of patients to process (for testing)')
    parser.add_argument('--output_dir', type=str, default='preprocessed_data',
                        help='Output directory for preprocessed data')
    
    args = parser.parse_args()
    
    preprocessor = MRIPreprocessor(output_dir=args.output_dir)
    preprocessor.run(max_patients=args.max_patients)
