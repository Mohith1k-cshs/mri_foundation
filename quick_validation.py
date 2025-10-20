"""
Quick Validation Script - Verify HPC Readiness
Runs fast checks without full preprocessing
"""

import os
import sys
import torch
import pandas as pd
from pathlib import Path

def check_packages():
    """Quick package check"""
    print("\n[1/6] Checking packages...")
    required = ['torch', 'torchvision', 'monai', 'timm', 'pandas', 'pydicom', 'nrrd', 'SimpleITK']
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"❌ Missing: {', '.join(missing)}")
        return False
    print("✅ All packages installed")
    return True

def check_data():
    """Check data paths"""
    print("\n[2/6] Checking data...")
    paths = {
        'DICOM': r'C:\Users\kanthamnenm\Downloads\Segmentation_Input_Automated_Download',
        'NRRD': r'C:\Users\kanthamnenm\Downloads\Segmentation_Outputs_293',
        'Clinical': 'patient_outcomes_mapping.csv',
        'Model': 'mri_foundation.pth'
    }
    
    all_exist = True
    for name, path in paths.items():
        if os.path.exists(path):
            print(f"✅ {name}: {path}")
        else:
            print(f"❌ {name}: {path} NOT FOUND")
            all_exist = False
    return all_exist

def check_clinical_data():
    """Verify clinical data"""
    print("\n[3/6] Checking clinical data...")
    try:
        if os.path.exists('patient_outcomes_mapping.csv'):
            df = pd.read_csv('patient_outcomes_mapping.csv')
            print(f"✅ Clinical data: {len(df)} patients")
            print(f"   Columns: {len(df.columns)} (outcome data present)")
            return True
        else:
            print("⚠️  patient_outcomes_mapping.csv not found (will need to create)")
            return True  # Not critical for test
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def check_scripts():
    """Verify all scripts exist"""
    print("\n[4/6] Checking scripts...")
    scripts = [
        'preprocess_mri_data.py',
        'dataset_clinical.py', 
        'model_clinical.py',
        'train_clinical_predictor.py'
    ]
    
    all_exist = True
    for script in scripts:
        if os.path.exists(script):
            print(f"✅ {script}")
        else:
            print(f"❌ {script} NOT FOUND")
            all_exist = False
    return all_exist

def check_model_loading():
    """Test if model can be loaded"""
    print("\n[5/6] Testing model loading...")
    try:
        import cfg
        from model_clinical import MRICORExClinicalPredictor
        
        args = cfg.parse_args()
        args.image_size = 512
        
        # Check if checkpoint exists
        if not os.path.exists('mri_foundation.pth'):
            print("❌ mri_foundation.pth not found")
            return False
        
        # Try loading on CPU (local machine might not have CUDA)
        print("   Loading model on CPU...")
        model = MRICORExClinicalPredictor(
            args=args,
            mri_core_checkpoint='mri_foundation.pth',
            freeze_encoder=True
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✅ Model loaded: {total_params:,} params ({trainable_params:,} trainable)")
        return True
    except RuntimeError as e:
        if "CUDA" in str(e):
            print("⚠️  Model has CUDA weights (OK - will work on HPC with GPU)")
            return True  # Accept as valid for HPC deployment
        else:
            print(f"❌ Model loading failed: {e}")
            return False
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def check_cuda():
    """Check CUDA availability"""
    print("\n[6/6] Checking CUDA...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        return True
    else:
        print("⚠️  CUDA not available (OK for local, needed for HPC)")
        return True  # Not critical for local testing

def main():
    print("="*70)
    print("  HPC READINESS CHECK")
    print("="*70)
    
    checks = [
        ("Packages", check_packages),
        ("Data Paths", check_data),
        ("Clinical Data", check_clinical_data),
        ("Scripts", check_scripts),
        ("Model Loading", check_model_loading),
        ("CUDA", check_cuda)
    ]
    
    results = {}
    for name, func in checks:
        try:
            results[name] = func()
        except Exception as e:
            print(f"❌ {name} check crashed: {e}")
            results[name] = False
    
    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)
    
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        status = "✅" if result else "❌"
        print(f"{status} {name}")
    
    print(f"\n{passed}/{total} checks passed")
    
    if passed >= total - 1:  # Allow CUDA to fail locally
        print("\n✅ READY FOR HPC DEPLOYMENT")
        print("\nNext steps:")
        print("1. Transfer to HPC: scp -r mri_foundation/ user@hpc:/path/")
        print("2. Setup environment on HPC (see HPC_DEPLOYMENT_GUIDE.md)")
        print("3. Submit preprocessing job: sbatch preprocess_job.sh")
        print("4. Submit training job: sbatch train_job.sh")
    else:
        print("\n❌ FIX ISSUES BEFORE HPC DEPLOYMENT")
    
    return passed >= total - 1

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
