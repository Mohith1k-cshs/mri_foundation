# HPC Deployment Options - Choose Your Method

## Option 1: Git Push & Clone (RECOMMENDED) ⭐

**Best for:** Version control, team collaboration, easy updates

### Advantages:
- ✅ Clean, professional workflow
- ✅ Easy to update code on HPC (just `git pull`)
- ✅ Version history preserved
- ✅ Team members can clone the same code
- ✅ Excludes large files automatically

### Steps:

#### Step 1: Commit and Push from Local Machine

```powershell
# Run this script
.\git_commit_and_push.ps1
```

Or manually:

```powershell
# Create .gitignore (exclude large files)
# See git_commit_and_push.ps1 for content

# Add files
git add *.py *.sh *.md requirements.txt patient_outcomes_mapping.csv models/ utils/ cfg.py

# Commit
git commit -m "Add HPC deployment scripts and complete pipeline"

# Push
git push origin branch_1
```

#### Step 2: Clone on HPC

```bash
# SSH to HPC
ssh your_username@hpc-address

# Clone repository
cd /home/your_username/projects
git clone https://github.com/Mohith1k-cshs/mri_foundation.git
cd mri_foundation
git checkout branch_1
```

#### Step 3: Transfer Large Files (Not in Git)

From your **local machine**, transfer files that are too large for Git:

```powershell
# Replace 'user@hpc' with your actual HPC login

# MRI-CORE pretrained model (416 MB)
scp mri_foundation.pth user@hpc:/home/user/projects/mri_foundation/

# DICOM data (if not already on HPC shared storage)
scp -r "C:\Users\kanthamnenm\Downloads\Segmentation_Input_Automated_Download" user@hpc:/data/mri/

# NRRD masks
scp -r "C:\Users\kanthamnenm\Downloads\Segmentation_Outputs_293" user@hpc:/data/mri/

# Clinical Excel file
scp "C:\Users\kanthamnenm\Downloads\Clinical_and_Other_Features.xlsx" user@hpc:/data/mri/
```

**Note:** If your DICOM/NRRD data is already on HPC shared storage, you only need to transfer the model file!

#### Step 4: Continue with HPC Setup

See **HPC_DEPLOYMENT_GUIDE.md** Step 3 (Setup Environment)

---

## Option 2: Direct SCP Transfer (Alternative)

**Best for:** Quick one-time transfer, no git experience needed

### Advantages:
- ✅ Simple, direct file transfer
- ✅ No git commands needed
- ✅ Everything in one transfer

### Disadvantages:
- ❌ No version control
- ❌ Hard to update later
- ❌ Transfers everything (large files too)

### Steps:

#### Method A: Using PowerShell (Windows)

```powershell
# Transfer entire directory
scp -r "C:\Users\kanthamnenm\Projects\mri_foundation" user@hpc:/home/user/projects/

# This will transfer:
# - All Python scripts
# - SLURM job scripts
# - Documentation
# - Model file (mri_foundation.pth)
# - patient_outcomes_mapping.csv
# - Everything except data files
```

#### Method B: Using WinSCP (GUI Tool)

1. Download WinSCP: https://winscp.net/
2. Connect to HPC
3. Drag and drop `mri_foundation` folder to HPC

#### Step 2: Transfer Data Files

```powershell
# If data not already on HPC
scp -r "C:\Users\kanthamnenm\Downloads\Segmentation_Input_Automated_Download" user@hpc:/data/mri/
scp -r "C:\Users\kanthamnenm\Downloads\Segmentation_Outputs_293" user@hpc:/data/mri/
scp "C:\Users\kanthamnenm\Downloads\Clinical_and_Other_Features.xlsx" user@hpc:/data/mri/
```

---

## Option 3: Hybrid Approach (Recommended if Data Already on HPC)

**Best for:** When your data is already on HPC shared storage

### Scenario:
Your DICOM/NRRD data might already be on HPC in a shared project directory like:
- `/data/breast_mri_study/`
- `/projects/shared_data/`
- `/scratch/group_name/mri_data/`

### Steps:

1. **Push code to Git** (Option 1, Steps 1-2)
2. **Only transfer the model file:**
   ```bash
   scp mri_foundation.pth user@hpc:/home/user/projects/mri_foundation/
   ```
3. **Update paths in scripts** to point to existing data location on HPC

---

## 🎯 Which Option Should You Choose?

### Choose **Option 1 (Git)** if:
- ✅ You plan to make updates to the code
- ✅ You want version control
- ✅ You work with a team
- ✅ You want professional workflow
- ✅ **RECOMMENDED for this project**

### Choose **Option 2 (SCP)** if:
- You need a quick one-time transfer
- You're unfamiliar with Git
- You won't be updating the code

### Choose **Option 3 (Hybrid)** if:
- Your data is already on HPC
- You want Git benefits but minimal transfer

---

## 📦 What Needs to Be Transferred?

### Code Files (Small - Use Git)
- ✅ `*.py` - All Python scripts
- ✅ `*.sh` - SLURM job scripts
- ✅ `*.md` - Documentation
- ✅ `requirements.txt` - Dependencies
- ✅ `patient_outcomes_mapping.csv` - Clinical labels (~100 KB)
- ✅ `models/` - Architecture code
- ✅ `utils/` - Utility functions
- ✅ `cfg.py` - Configuration

**Total size: ~500 KB** (perfect for Git)

### Large Files (Use SCP)
- 📦 `mri_foundation.pth` - **416 MB** (pretrained model)
- 📦 DICOM data - **~10-20 GB** (raw images)
- 📦 NRRD masks - **~1-2 GB** (segmentation masks)
- 📦 `Clinical_and_Other_Features.xlsx` - **0.6 MB** (Excel file)

**Total size: ~12-23 GB**

---

## 🚀 Quick Start Commands

### If Using Git (Recommended):

**Local Machine:**
```powershell
# 1. Commit and push
.\git_commit_and_push.ps1

# 2. Transfer large files
scp mri_foundation.pth user@hpc:/home/user/projects/mri_foundation/
```

**On HPC:**
```bash
# 3. Clone
git clone https://github.com/Mohith1k-cshs/mri_foundation.git
cd mri_foundation
git checkout branch_1

# 4. Setup environment (see HPC_DEPLOYMENT_GUIDE.md)
module load python/3.11 cuda/12.4
python -m venv mri_env
source mri_env/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# 5. Submit jobs
chmod +x *.sh
sbatch preprocess_job.sh
```

---

## 🔍 Checking Data Location on HPC

Before transferring large data files, check if they're already on HPC:

```bash
# SSH to HPC
ssh user@hpc-address

# Common data locations to check:
ls /data/
ls /projects/
ls /scratch/
ls /gpfs/

# Look for breast MRI study data
find /data -name "*Breast_MRI*" -type d 2>/dev/null | head -20
find /projects -name "*Segmentation*" -type d 2>/dev/null | head -20

# If you find it, note the path and update preprocess_mri_data.py accordingly
```

---

## 📝 Summary

**My Recommendation for You:**

1. ✅ **Use Git (Option 1)** - Professional and maintainable
2. ✅ **Run:** `.\git_commit_and_push.ps1`
3. ✅ **Transfer only:** `mri_foundation.pth` via SCP
4. ✅ **Check if data already on HPC** before transferring 20+ GB
5. ✅ **Follow:** HPC_DEPLOYMENT_GUIDE.md for setup

**Time Estimate:**
- Git push: 2-3 minutes
- Transfer model (416 MB): 5-10 minutes
- Transfer data (if needed): 1-2 hours
- HPC setup: 15-20 minutes

**Total: 20-30 minutes** (or 2-3 hours if transferring all data)

Ready to start? Run `.\git_commit_and_push.ps1` now!
