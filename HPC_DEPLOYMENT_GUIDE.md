# HPC Deployment Guide for MRI Clinical Predictor

## 📋 Overview

This guide helps you deploy the MRI-CORE clinical predictor on your HPC cluster (NVIDIA DGX Spark).

**Time Estimates for HPC:**
- **Preprocessing**: 1.5-3 hours (CPU, 8 cores)
- **Training (Single A100)**: 4-6 hours 
- **Training (4x A100)**: 2-3 hours (with multi-GPU setup)
- **Evaluation**: 5-10 minutes

---

## 🧪 STEP 0: Test Locally First (CRITICAL!)

**Before transferring to HPC, verify everything works:**

```powershell
# Run comprehensive pipeline test
python test_pipeline.py
```

This will test:
- ✅ All packages installed
- ✅ Data paths exist
- ✅ Clinical data loads correctly
- ✅ Preprocessing works (3 patients)
- ✅ Dataset can load data
- ✅ Model can be instantiated
- ✅ Training script is valid

**IMPORTANT:** Fix any failed tests before moving to HPC!

---

## 📦 STEP 1: Prepare for HPC Transfer

### 1.1 Create requirements file for HPC

```powershell
# Generate exact package versions
pip freeze > requirements_hpc.txt
```

### 1.2 Package your code

```powershell
# Create deployment directory
mkdir mri_foundation_hpc
cd mri_foundation_hpc

# Copy necessary files
Copy-Item -Recurse ..\models .
Copy-Item -Recurse ..\utils .
Copy-Item ..\*.py .
Copy-Item ..\*.sh .
Copy-Item ..\requirements.txt .
Copy-Item ..\mri_foundation.pth .
Copy-Item ..\README.md .

# Create logs directory
mkdir logs
```

### 1.3 Create data manifest

```powershell
# Document your data locations
@"
Data Locations for HPC:
========================

DICOM Input: C:\Users\kanthamnenm\Downloads\Segmentation_Input_Automated_Download
  - ~292 patient folders
  - ~43,800 slices total

NRRD Masks: C:\Users\kanthamnenm\Downloads\Segmentation_Outputs_293
  - Segmentation_Breast_MRI_XXX_Breast.seg.nrrd files

Clinical Data: C:\Users\kanthamnenm\Downloads\Clinical_and_Other_Features.xlsx
  - 923 patients, 292 NAC patients

Pretrained Model: mri_foundation.pth (included in transfer)
"@ | Out-File -FilePath data_manifest.txt
```

---

## 🚀 STEP 2: Transfer to HPC

### 2.1 Transfer code and model

```bash
# From your local machine to HPC
scp -r mri_foundation_hpc/ username@hpc-address:/home/username/projects/

# Transfer data (if not already on HPC shared storage)
scp -r Segmentation_Input_Automated_Download/ username@hpc-address:/data/mri/
scp -r Segmentation_Outputs_293/ username@hpc-address:/data/mri/
scp Clinical_and_Other_Features.xlsx username@hpc-address:/data/mri/
```

### 2.2 SSH to HPC

```bash
ssh username@hpc-address
cd /home/username/projects/mri_foundation_hpc
```

---

## 🐍 STEP 3: Setup Environment on HPC

### 3.1 Create virtual environment

```bash
# Load Python module (adjust for your HPC)
module load python/3.11

# Create virtual environment
python -m venv mri_env
source mri_env/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 3.2 Install PyTorch with CUDA

```bash
# Install PyTorch for CUDA 12.4 (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 3.3 Install other dependencies

```bash
# Install all required packages
pip install monai==1.5.0
pip install timm==1.0.20
pip install einops==0.8.1
pip install scikit-learn==1.7.2
pip install pandas==2.3.3
pip install seaborn==0.13.2
pip install matplotlib==3.10.7
pip install pydicom
pip install pynrrd
pip install SimpleITK
pip install openpyxl
```

### 3.4 Verify GPU access

```bash
# Check CUDA
module load cuda/12.4
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPUs: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
```

Expected output:
```
CUDA: True
GPUs: 4
GPU 0: NVIDIA A100-SXM4-80GB
GPU 1: NVIDIA A100-SXM4-80GB
GPU 2: NVIDIA A100-SXM4-80GB
GPU 3: NVIDIA A100-SXM4-80GB
```

---

## 🔧 STEP 4: Update Data Paths

### 4.1 Update paths in preprocessing script

```bash
# Edit preprocess_mri_data.py
nano preprocess_mri_data.py
```

Update the paths at the bottom:
```python
if __name__ == "__main__":
    preprocessor = MRIPreprocessor(
        dicom_root='/data/mri/Segmentation_Input_Automated_Download',  # Update this
        nrrd_root='/data/mri/Segmentation_Outputs_293',                # Update this
        clinical_csv='/data/mri/Clinical_and_Other_Features.xlsx',     # Update this
        output_dir='preprocessed_data',
        test_mode=False  # Set to False for full processing
    )
```

---

## 🧪 STEP 5: Run Small Test on HPC

**Before submitting big jobs, test with interactive session:**

```bash
# Request interactive GPU node
salloc --time=00:30:00 --cpus-per-task=4 --mem=16G --gres=gpu:1

# Activate environment
source mri_env/bin/activate

# Test preprocessing on 3 patients
python -c "
from preprocess_mri_data import MRIPreprocessor
preprocessor = MRIPreprocessor(
    dicom_root='/data/mri/Segmentation_Input_Automated_Download',
    nrrd_root='/data/mri/Segmentation_Outputs_293',
    clinical_csv='/data/mri/Clinical_and_Other_Features.xlsx',
    output_dir='test_preprocessed',
    test_mode=True
)
results = preprocessor.process_all_patients()
print(f'Success! Processed {results[\"successful_patients\"]} patients, {results[\"total_slices\"]} slices')
"

# Test model loading
python -c "
import torch
from model_clinical import MRICORExClinicalPredictor
model = MRICORExClinicalPredictor('mri_foundation.pth', freeze_encoder=True)
print(f'Model loaded! CUDA: {next(model.parameters()).is_cuda}')
dummy = torch.randn(2, 3, 512, 512)
outputs = model(dummy)
print('Forward pass successful!')
"

# Exit interactive session
exit
```

---

## 🎯 STEP 6: Submit Production Jobs

### 6.1 Make scripts executable

```bash
chmod +x preprocess_job.sh
chmod +x train_job.sh
chmod +x train_job_multi_gpu.sh
```

### 6.2 Create logs directory

```bash
mkdir -p logs
```

### 6.3 Submit preprocessing job

```bash
# Submit to SLURM scheduler
sbatch preprocess_job.sh

# Check status
squeue -u $USER

# Monitor logs (in another terminal)
tail -f logs/preprocess_*.out
```

**What it does:**
- Processes all ~292 NAC patients
- Converts DICOM → PNG
- Extracts clinical labels
- Saves to `preprocessed_data/`
- **Time:** 2-4 hours

### 6.4 Wait for preprocessing to complete

```bash
# Check if preprocessing finished
ls -lh preprocessed_data/metadata.csv

# Count slices
wc -l preprocessed_data/metadata.csv
# Expected: ~43,800 slices

# Check disk usage
du -sh preprocessed_data/
# Expected: ~2-3 GB
```

### 6.5 Submit training job

**Option A: Single GPU (Recommended)**
```bash
sbatch train_job.sh
```
- Uses 1x A100 GPU
- Batch size: 32
- Time: 4-6 hours

**Option B: Multi-GPU (If you want faster training)**
```bash
sbatch train_job_multi_gpu.sh
```
- Uses 4x A100 GPUs
- Batch size: 64
- Time: 2-3 hours
- **Note:** Requires multi-GPU support in training script (see Step 7)

### 6.6 Monitor training

```bash
# Check job status
squeue -u $USER

# Watch training logs
tail -f logs/train_*.out

# Check GPU usage (from the compute node)
ssh compute-node-name  # Get node name from squeue
watch -n 1 nvidia-smi
```

---

## 📊 STEP 7: Retrieve and Analyze Results

### 7.1 Check if training completed

```bash
# Check for output files
ls -lh outputs/clinical_predictor/

# Should see:
# - best_model.pth
# - results.json
# - training_curves.png
# - confusion_matrices.png
```

### 7.2 View metrics

```bash
# Print results
cat outputs/clinical_predictor/results.json

# Or formatted
python -c "
import json
results = json.load(open('outputs/clinical_predictor/results.json'))
print('\nTest Metrics:')
for task, metrics in results['test_metrics'].items():
    print(f'{task}: {metrics}')
"
```

### 7.3 Download results to local machine

```bash
# From your local machine
scp username@hpc-address:/home/username/projects/mri_foundation_hpc/outputs/clinical_predictor/* ./results/
```

### 7.4 View plots locally

```powershell
# Open results folder
explorer .\results\

# View:
# - training_curves.png (loss and metrics over time)
# - confusion_matrices.png (error analysis per task)
```

---

## 🔍 Troubleshooting HPC Issues

### Issue 1: "CUDA out of memory"

**Solution:** Reduce batch size in `train_job.sh`:
```bash
BATCH_SIZE=16  # or 8
```

### Issue 2: "No module named 'torch'"

**Solution:** Ensure virtual environment is activated:
```bash
source mri_env/bin/activate
which python  # Should show /path/to/mri_env/bin/python
```

### Issue 3: "FileNotFoundError: clinical data"

**Solution:** Update paths in scripts to match your HPC storage:
```bash
# Check where your data is
ls /data/mri/
ls /scratch/username/

# Update paths in preprocess_mri_data.py
```

### Issue 4: Job timeout

**Solution:** Increase time limit in job scripts:
```bash
#SBATCH --time=24:00:00  # 24 hours
```

### Issue 5: "Preprocessing found 0 NAC patients"

**Solution:** Check NAC column comparison in `preprocess_mri_data.py`:
```python
# Make sure it's comparing strings
nac_patients = df[df['received_nac'].astype(str) == '1']
```

---

## 📈 Expected Results

### Preprocessing Output
- **Patients processed:** ~292
- **Total slices:** ~43,800
- **Disk space:** ~2-3 GB
- **Files:** 
  - `preprocessed_data/images/*.png`
  - `preprocessed_data/masks/*.png`
  - `preprocessed_data/metadata.csv`

### Training Output
- **Best model:** `outputs/clinical_predictor/best_model.pth` (~350 MB)
- **Target metrics:**
  - pT accuracy: > 0.60
  - pN accuracy: > 0.65
  - pM accuracy: > 0.90
  - **pCR accuracy: > 0.70** (most important)
  - pCR F1-score: > 0.70

---

## 🎯 Quick Reference Commands

### Check job status
```bash
squeue -u $USER              # Your jobs
squeue -j <job_id>           # Specific job
sacct -j <job_id>            # Job history
```

### Cancel jobs
```bash
scancel <job_id>             # Cancel specific job
scancel -u $USER             # Cancel all your jobs
```

### Check resources
```bash
sinfo                        # Partition info
sinfo -p gpu --long          # GPU partition details
scontrol show partition gpu  # Detailed partition info
```

### Monitor logs
```bash
tail -f logs/preprocess_*.out    # Watch preprocessing
tail -f logs/train_*.out         # Watch training
less logs/preprocess_*.err       # Check errors
```

### Disk usage
```bash
du -sh preprocessed_data/        # Check preprocessing output
du -sh outputs/                  # Check training output
quota -s                         # Your storage quota
```

---

## ✅ Checklist

**Before HPC transfer:**
- [ ] Run `test_pipeline.py` locally - all tests pass
- [ ] Package code and model
- [ ] Document data locations

**On HPC:**
- [ ] Transfer code and data
- [ ] Create virtual environment
- [ ] Install dependencies
- [ ] Verify GPU access with `nvidia-smi`
- [ ] Update data paths in scripts
- [ ] Run small interactive test (3 patients)
- [ ] Submit preprocessing job (`sbatch preprocess_job.sh`)
- [ ] Wait for preprocessing (2-4 hours)
- [ ] Verify preprocessing output (~43,800 slices)
- [ ] Submit training job (`sbatch train_job.sh`)
- [ ] Monitor training (4-6 hours)
- [ ] Download results
- [ ] Analyze metrics and plots

---

## 📧 Getting Help

If you encounter issues:
1. Check error logs: `logs/*.err`
2. Check output logs: `logs/*.out`
3. Verify environment: `which python`, `python -c "import torch"`
4. Check data paths: `ls /data/mri/`
5. Test interactively: `salloc` then run commands manually

**Common HPC module names to try:**
```bash
module avail python
module avail cuda
module avail cudnn
```
