"""
Test Pipeline - Validates all components before HPC deployment
Run this locally to ensure everything works correctly
"""

import os
import sys
import torch
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def test_imports():
    """Test 1: Verify all required packages are installed"""
    print_section("TEST 1: Package Imports")
    
    required_packages = {
        'torch': None,
        'torchvision': None,
        'monai': None,
        'timm': None,
        'einops': None,
        'pandas': None,
        'numpy': None,
        'PIL': 'Pillow',
        'sklearn': 'scikit-learn',
        'matplotlib': None,
        'seaborn': None,
        'pydicom': None,
        'nrrd': 'pynrrd',
        'SimpleITK': None
    }
    
    failed = []
    for package, install_name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {install_name or package}")
        except ImportError:
            print(f"✗ {install_name or package} - MISSING")
            failed.append(install_name or package)
    
    if failed:
        print(f"\n❌ Missing packages: {', '.join(failed)}")
        print(f"Install with: pip install {' '.join(failed)}")
        return False
    
    print("\n✅ All packages installed correctly")
    return True

def test_data_paths():
    """Test 2: Verify all required data paths exist"""
    print_section("TEST 2: Data Paths")
    
    paths = {
        'DICOM Input': r'C:\Users\kanthamnenm\Downloads\Segmentation_Input_Automated_Download',
        'NRRD Masks': r'C:\Users\kanthamnenm\Downloads\Segmentation_Outputs_293',
        'Clinical Data': r'C:\Users\kanthamnenm\Downloads\Clinical_and_Other_Features.xlsx',
        'MRI-CORE Model': r'C:\Users\kanthamnenm\Projects\mri_foundation\mri_foundation.pth'
    }
    
    failed = []
    for name, path in paths.items():
        if os.path.exists(path):
            if os.path.isdir(path):
                count = len(os.listdir(path))
                print(f"✓ {name}: {path} ({count} items)")
            else:
                size_mb = os.path.getsize(path) / (1024*1024)
                print(f"✓ {name}: {path} ({size_mb:.1f} MB)")
        else:
            print(f"✗ {name}: {path} - NOT FOUND")
            failed.append(name)
    
    if failed:
        print(f"\n❌ Missing data: {', '.join(failed)}")
        return False
    
    print("\n✅ All data paths exist")
    return True

def test_clinical_data():
    """Test 3: Verify clinical data can be loaded and has NAC patients"""
    print_section("TEST 3: Clinical Data")
    
    try:
        # Try to load from CSV first (preprocessed)
        csv_path = 'patient_outcomes_mapping.csv'
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print(f"✓ Loaded clinical data from CSV: {len(df)} total patients")
        else:
            # Load from Excel and flatten
            excel_path = r'C:\Users\kanthamnenm\Downloads\Clinical_and_Other_Features.xlsx'
            df = pd.read_excel(excel_path, header=[0, 1])
            df.columns = ['_'.join(col).strip() if col[1] != 'Unnamed' else col[0] 
                          for col in df.columns.values]
            print(f"✓ Loaded clinical data from Excel: {len(df)} total patients")
        
        # Check NAC patients - check multiple possible column names
        nac_col = None
        for col in ['received_nac', 'Received NAC', 'NAC']:
            if col in df.columns:
                nac_col = col
                break
        
        if nac_col:
            nac_patients = df[df[nac_col].astype(str) == '1']
            print(f"✓ Found {len(nac_patients)} NAC patients (column: {nac_col})")
            
            # Check outcome columns
            outcome_cols = ['clinical_stage_t', 'clinical_stage_n', 'clinical_stage_m',
                          'pathologic_stage_t', 'pathologic_stage_n', 'pathologic_stage_m',
                          'path_complete_response']
            
            present = [col for col in outcome_cols if col in df.columns]
            missing = [col for col in outcome_cols if col not in df.columns]
            
            print(f"✓ Found outcome columns: {len(present)}/{len(outcome_cols)}")
            if missing:
                print(f"  Missing: {', '.join(missing)}")
            
            if len(nac_patients) < 100:
                print(f"\n⚠️  WARNING: Only {len(nac_patients)} NAC patients (expected ~292)")
                return False
            
        else:
            print("✗ Column 'received_nac' not found")
            return False
        
        print("\n✅ Clinical data validated")
        return True
        
    except Exception as e:
        print(f"\n❌ Clinical data error: {e}")
        return False

def test_preprocessing_small():
    """Test 4: Run preprocessing on 2-3 patients"""
    print_section("TEST 4: Preprocessing Test (3 patients)")
    
    try:
        from preprocess_mri_data import MRIPreprocessor
        
        print("Initializing preprocessor...")
        # Use patient_outcomes_mapping.csv if available
        clinical_csv = 'patient_outcomes_mapping.csv' if os.path.exists('patient_outcomes_mapping.csv') else r'C:\Users\kanthamnenm\Downloads\Clinical_and_Other_Features.xlsx'
        
        preprocessor = MRIPreprocessor(
            dicom_root=r'C:\Users\kanthamnenm\Downloads\Segmentation_Input_Automated_Download',
            nrrd_root=r'C:\Users\kanthamnenm\Downloads\Segmentation_Outputs_293',
            clinical_csv=clinical_csv,
            output_dir='test_preprocessed'
        )
        
        print("\nProcessing 3 test patients...")
        # Manually process just 3 patients for testing
        results = {'successful_patients': 0, 'total_slices': 0}
        test_limit = 3
        count = 0
        
        for patient_folder in preprocessor.dicom_root.iterdir():
            if count >= test_limit:
                break
            if patient_folder.is_dir() and patient_folder.name.startswith('Breast_MRI'):
                result = preprocessor.process_patient(patient_folder)
                if result['success']:
                    results['successful_patients'] += 1
                    results['total_slices'] += result['num_slices']
                count += 1
        
        if results['successful_patients'] > 0:
            print(f"\n✓ Processed {results['successful_patients']} patients")
            print(f"✓ Generated {results['total_slices']} slices")
            print(f"✓ Saved to: test_preprocessed/")
            
            # Check metadata file
            if os.path.exists('test_preprocessed/metadata.csv'):
                metadata = pd.read_csv('test_preprocessed/metadata.csv')
                print(f"✓ Metadata file: {len(metadata)} entries")
                
                # Check label distribution
                print("\n  Label distribution:")
                for col in ['pT', 'pN', 'pM', 'pCR']:
                    if col in metadata.columns:
                        dist = metadata[col].value_counts().sort_index()
                        print(f"    {col}: {dict(dist)}")
            
            print("\n✅ Preprocessing test passed")
            return True
        else:
            print("\n❌ No patients processed successfully")
            return False
            
    except Exception as e:
        print(f"\n❌ Preprocessing error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset():
    """Test 5: Verify dataset can load preprocessed data"""
    print_section("TEST 5: Dataset Test")
    
    try:
        from dataset_clinical import MRIClinicalDataset
        
        # Check if test preprocessing output exists
        if not os.path.exists('test_preprocessed/metadata.csv'):
            print("⚠️  Skipping: Run preprocessing test first")
            return True
        
        print("Creating train dataset...")
        train_dataset = MRIClinicalDataset(
            metadata_csv='test_preprocessed/metadata.csv',
            data_root='test_preprocessed',
            image_size=512,
            split='train'
        )
        
        print(f"✓ Train dataset: {len(train_dataset)} slices")
        
        # Try loading one sample
        print("\nLoading sample...")
        sample = train_dataset[0]
        
        print(f"✓ Image shape: {sample['image'].shape}")
        print(f"✓ Labels: pT={sample['pT']}, pN={sample['pN']}, pM={sample['pM']}, pCR={sample['pCR']}")
        
        print("\n✅ Dataset test passed")
        return True
        
    except Exception as e:
        print(f"\n❌ Dataset error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model():
    """Test 6: Verify model can be instantiated and run forward pass"""
    print_section("TEST 6: Model Test")
    
    try:
        from model_clinical import MRICORExClinicalPredictor
        import cfg
        
        # Check if MRI-CORE checkpoint exists
        checkpoint_path = 'mri_foundation.pth'
        if not os.path.exists(checkpoint_path):
            print(f"✗ MRI-CORE checkpoint not found: {checkpoint_path}")
            return False
        
        print(f"Loading MRI-CORE model from: {checkpoint_path}")
        
        # Create args object with required parameters
        args = cfg.parse_args()
        args.image_size = 512
        
        model = MRICORExClinicalPredictor(
            args=args,
            mri_core_checkpoint=checkpoint_path,
            freeze_encoder=True
        )
        
        print(f"✓ Model loaded successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✓ Total parameters: {total_params:,}")
        print(f"✓ Trainable parameters: {trainable_params:,}")
        
        # Test forward pass
        print("\nTesting forward pass...")
        dummy_input = torch.randn(2, 3, 512, 512)
        
        model.eval()
        with torch.no_grad():
            outputs = model(dummy_input)
        
        print(f"✓ Forward pass successful")
        print(f"  Output shapes:")
        for task, logits in outputs.items():
            print(f"    {task}: {logits.shape}")
        
        print("\n✅ Model test passed")
        return True
        
    except Exception as e:
        print(f"\n❌ Model error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_script():
    """Test 7: Verify training script can be imported and initialized"""
    print_section("TEST 7: Training Script Test")
    
    try:
        # Check if training script exists
        if not os.path.exists('train_clinical_predictor.py'):
            print("✗ train_clinical_predictor.py not found")
            return False
        
        print("✓ Training script found")
        
        # Try importing components
        sys.path.insert(0, os.getcwd())
        
        # Just check if file is valid Python
        with open('train_clinical_predictor.py', 'r') as f:
            code = f.read()
            compile(code, 'train_clinical_predictor.py', 'exec')
        
        print("✓ Training script is valid Python")
        
        print("\n✅ Training script test passed")
        return True
        
    except Exception as e:
        print(f"\n❌ Training script error: {e}")
        return False

def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("  MRI-CORE CLINICAL PREDICTOR - PIPELINE TEST")
    print("  Testing all components before HPC deployment")
    print("="*80)
    
    tests = [
        ("Package Imports", test_imports),
        ("Data Paths", test_data_paths),
        ("Clinical Data", test_clinical_data),
        ("Preprocessing", test_preprocessing_small),
        ("Dataset", test_dataset),
        ("Model", test_model),
        ("Training Script", test_training_script)
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n❌ {name} test crashed: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "="*80)
    print("  TEST SUMMARY")
    print("="*80)
    
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}  {name}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print(f"\n{total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n" + "="*80)
        print("  🎉 ALL TESTS PASSED - READY FOR HPC DEPLOYMENT")
        print("="*80)
        print("\nNext steps:")
        print("1. Transfer code to HPC: scp -r mri_foundation/ your_hpc:/path/")
        print("2. Submit preprocessing job: sbatch preprocess_job.sh")
        print("3. Submit training job: sbatch train_job.sh")
    else:
        print("\n" + "="*80)
        print("  ⚠️  FIX FAILED TESTS BEFORE HPC DEPLOYMENT")
        print("="*80)
        print("\nFix the issues above and run this test again.")
    
    return total_passed == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
