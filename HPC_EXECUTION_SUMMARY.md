# HPC Execution Summary - MRI Clinical Predictor

## ✅ Your System is Ready for HPC Deployment!

**Quick Validation Status:** 6/6 checks passed

---

## 📊 Time Estimates for NVIDIA DGX Spark

Based on your **NVIDIA DGX Spark** system (4x A100 80GB GPUs):

| Step | Task | Hardware | Time | Output |
|------|------|----------|------|--------|
| 1 | **Preprocessing** | 8 CPU cores | **1.5-3 hours** | ~43,800 PNG slices (~2-3 GB) |
| 2 | **Training (Single GPU)** | 1x A100 80GB | **4-6 hours** | Trained model + metrics |
| 2 | **Training (Multi-GPU)** | 4x A100 80GB | **2-3 hours** | Trained model + metrics (faster) |
| 3 | **Evaluation** | 1x A100 | **5-10 minutes** | Results, plots, confusion matrices |

**Total End-to-End Time:** 
- With single GPU: **6-9 hours**
- With 4 GPUs: **4-6 hours**

---

## 🚀 Quick Start Guide

### Step 1: Before Transferring to HPC

✅ **Validation Complete** - You've verified locally that:
- All packages are installed
- All data paths exist
- Scripts are valid
- Model can be loaded

### Step 2: Transfer to HPC

```bash
# Package your code
cd C:\Users\kanthamnenm\Projects\mri_foundation

# Create deployment package (if needed)
# Then from your local machine, transfer to HPC:

scp -r mri_foundation/ username@hpc-address:/home/username/projects/
scp -r C:/Users/kanthamnenm/Downloads/Segmentation_Input_Automated_Download username@hpc-address:/data/mri/
scp -r C:/Users/kanthamnenm/Downloads/Segmentation_Outputs_293 username@hpc-address:/data/mri/
scp C:/Users/kanthamnenm/Downloads/Clinical_and_Other_Features.xlsx username@hpc-address:/data/mri/
```

### Step 3: Setup on HPC

```bash
# SSH to HPC
ssh username@hpc-address
cd /home/username/projects/mri_foundation

# Load modules (adjust for your HPC)
module load python/3.11
module load cuda/12.4

# Create virtual environment
python -m venv mri_env
source mri_env/bin/activate

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install dependencies
pip install monai==1.5.0 timm==1.0.20 einops==0.8.1 scikit-learn pandas seaborn matplotlib pydicom pynrrd SimpleITK openpyxl

# Verify GPU
nvidia-smi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

### Step 4: Update Data Paths

Edit `preprocess_mri_data.py` (bottom of file):

```python
if __name__ == "__main__":
    preprocessor = MRIPreprocessor(
        dicom_root='/data/mri/Segmentation_Input_Automated_Download',  # Update this
        nrrd_root='/data/mri/Segmentation_Outputs_293',                # Update this
        clinical_csv='/data/mri/Clinical_and_Other_Features.xlsx',     # Update this
        output_dir='preprocessed_data',
        test_mode=False  # Full processing
    )
```

### Step 5: Quick Test on HPC

```bash
# Request interactive session
salloc --time=00:30:00 --cpus-per-task=4 --mem=16G --gres=gpu:1

# Activate environment
source mri_env/bin/activate

# Run quick validation
python quick_validation.py

# Should see: "✅ READY FOR HPC DEPLOYMENT"
exit
```

### Step 6: Submit Jobs

```bash
# Make scripts executable
chmod +x preprocess_job.sh train_job.sh train_job_multi_gpu.sh

# Create logs directory
mkdir -p logs

# Submit preprocessing (1.5-3 hours)
sbatch preprocess_job.sh

# Monitor
squeue -u $USER
tail -f logs/preprocess_*.out

# Wait for completion, then verify
wc -l preprocessed_data/metadata.csv  # Should show ~43,800

# Submit training
sbatch train_job.sh           # Single A100 (4-6 hours)
# OR
sbatch train_job_multi_gpu.sh # 4x A100 (2-3 hours, faster)

# Monitor training
tail -f logs/train_*.out
```

### Step 7: Retrieve Results

```bash
# Check outputs
ls -lh outputs/clinical_predictor/

# View metrics
cat outputs/clinical_predictor/results.json

# Download to local machine (from local terminal)
scp username@hpc-address:/home/username/projects/mri_foundation/outputs/clinical_predictor/* ./results/
```

---

## 📁 Key Files for HPC

### Scripts to Transfer
- `preprocess_mri_data.py` - DICOM → PNG preprocessing
- `dataset_clinical.py` - PyTorch dataset class
- `model_clinical.py` - MRI-CORE + clinical heads
- `train_clinical_predictor.py` - Training loop
- `preprocess_job.sh` - SLURM job for preprocessing
- `train_job.sh` - SLURM job for single-GPU training
- `train_job_multi_gpu.sh` - SLURM job for multi-GPU training
- `quick_validation.py` - Validation script
- `mri_foundation.pth` - Pretrained MRI-CORE model (416 MB)
- `patient_outcomes_mapping.csv` - Clinical labels
- `models/` - Model architecture directory
- `utils/` - Utility functions directory

### Documentation
- `HPC_DEPLOYMENT_GUIDE.md` - Comprehensive HPC guide
- `COMPLETE_GUIDE.md` - Full pipeline documentation
- `FINAL_SUMMARY.md` - Project summary

---

## 🎯 Expected Outputs

### After Preprocessing
```
preprocessed_data/
├── images/
│   └── *.png (~43,800 files)
├── masks/
│   └── *.png (~43,800 files)
└── metadata.csv (patient_id, slice_idx, pT, pN, pM, pCR labels)
```

### After Training
```
outputs/clinical_predictor/
├── best_model.pth (~350 MB)
├── results.json (test metrics)
├── training_curves.png (loss & accuracy plots)
└── confusion_matrices.png (per-task error analysis)
```

### Target Metrics
- **pT accuracy:** > 0.60
- **pN accuracy:** > 0.65
- **pM accuracy:** > 0.90
- **pCR accuracy:** > 0.70 ⭐ (most important)
- **pCR F1-score:** > 0.70

---

## 🔍 Monitoring Jobs

### Check Job Status
```bash
squeue -u $USER              # Your jobs
squeue -j <job_id>           # Specific job details
sacct -j <job_id> --format=JobID,JobName,State,ExitCode,Elapsed
```

### View Logs
```bash
# Watch live
tail -f logs/preprocess_*.out
tail -f logs/train_*.out

# Check errors
less logs/preprocess_*.err
less logs/train_*.err
```

### GPU Monitoring (from compute node)
```bash
# Get node name from squeue
squeue -u $USER

# SSH to compute node
ssh compute-node-name

# Watch GPU usage
watch -n 1 nvidia-smi
```

---

## ⚠️ Common Issues & Solutions

### Issue: "CUDA out of memory"
**Solution:** Reduce batch size in `train_job.sh`:
```bash
BATCH_SIZE=16  # or 8 if still failing
```

### Issue: "FileNotFoundError: Clinical data"
**Solution:** Verify paths in `preprocess_mri_data.py` match your HPC storage

### Issue: "No patients processed"
**Solution:** Check NAC filtering - ensure clinical CSV is loaded correctly

### Issue: Job timeout
**Solution:** Increase time in job script:
```bash
#SBATCH --time=24:00:00
```

---

## 📊 Training Configuration

### Recommended for Single A100 (80GB)
```bash
BATCH_SIZE=32
IMAGE_SIZE=512
EPOCHS=100
LEARNING_RATE=0.0001
FREEZE_ENCODER=true
```

### Recommended for 4x A100 (Multi-GPU)
```bash
BATCH_SIZE=64  # per GPU effective = 256
IMAGE_SIZE=512
EPOCHS=100
LEARNING_RATE=0.0001
```

---

## ✅ Verification Checklist

**Before Submitting Jobs:**
- [ ] Transferred all code and data to HPC
- [ ] Created virtual environment with all dependencies
- [ ] Verified GPU access: `nvidia-smi` shows 4x A100
- [ ] Updated data paths in `preprocess_mri_data.py`
- [ ] Ran `python quick_validation.py` - all checks passed
- [ ] Created `logs/` directory
- [ ] Made job scripts executable: `chmod +x *.sh`

**After Preprocessing:**
- [ ] Check metadata: `wc -l preprocessed_data/metadata.csv` shows ~43,800
- [ ] Check disk: `du -sh preprocessed_data/` shows ~2-3 GB
- [ ] Verify images: `ls preprocessed_data/images/*.png | wc -l` shows ~43,800

**After Training:**
- [ ] Check model: `ls -lh outputs/clinical_predictor/best_model.pth` shows ~350 MB
- [ ] Review metrics: `cat outputs/clinical_predictor/results.json`
- [ ] Check pCR accuracy: Should be > 0.70
- [ ] Download results to local machine
- [ ] Analyze confusion matrices and training curves

---

## 📞 Quick Reference

### Submit Jobs
```bash
sbatch preprocess_job.sh      # Step 1: Preprocessing
sbatch train_job.sh           # Step 2: Training (single GPU)
sbatch train_job_multi_gpu.sh # Step 2: Training (4 GPUs)
```

### Cancel Jobs
```bash
scancel <job_id>    # Cancel specific job
scancel -u $USER    # Cancel all your jobs
```

### Check Resources
```bash
sinfo -p gpu --long        # GPU partition info
squeue -p gpu              # Jobs in GPU queue
```

---

## 🎓 What This Pipeline Does

1. **Preprocessing:** Converts raw DICOM breast MRI volumes → 2D PNG slices with clinical outcome labels (pT, pN, pM, pCR)

2. **Training:** Uses pretrained MRI-CORE foundation model (ViT-B encoder) + custom multi-task prediction heads to predict post-chemotherapy pathological outcomes from pre-treatment MRI

3. **Evaluation:** Generates test metrics, confusion matrices, and training curves to assess clinical prediction accuracy

**Clinical Goal:** Predict neoadjuvant chemotherapy response (pCR) and TNM staging from pre-treatment MRI to aid treatment planning for breast cancer patients

---

## 📈 What Success Looks Like

**Excellent Results (pCR accuracy > 0.75):**
- Can reliably identify complete responders
- Useful for treatment planning and patient counseling
- Consider publication

**Good Results (pCR accuracy 0.70-0.75):**
- Clinically relevant predictions
- May benefit from fine-tuning

**Moderate Results (pCR accuracy 0.60-0.70):**
- Better than random, needs improvement
- Try unfreezing encoder or adding clinical features

**Below Target (pCR accuracy < 0.60):**
- Needs significant improvement
- Check data quality, class balance, model architecture

---

## 🚀 Ready to Start!

You have everything you need:
- ✅ Code validated locally
- ✅ All dependencies listed
- ✅ HPC job scripts ready
- ✅ Comprehensive documentation

**Next immediate action:** Transfer code and data to HPC, then follow Step 3 in this guide.

For detailed instructions, see:
- **HPC_DEPLOYMENT_GUIDE.md** - Step-by-step HPC setup
- **COMPLETE_GUIDE.md** - Full technical documentation
- **FINAL_SUMMARY.md** - Project overview

Good luck with your HPC run! 🎯
