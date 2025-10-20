# MRI-CORE Clinical Prediction Pipeline - Progress Report

## 📋 Project Overview

**Objective**: Train MRI-CORE foundation model to predict post-chemotherapy pathological outcomes from pre-treatment breast MRI scans.

**Clinical Question**: Can pre-treatment MRI features predict response to neoadjuvant chemotherapy (NAC)?

**Target Outcomes**:
- **pT** (Pathologic Tumor Stage): T0, T1, T2, T3, T4, Tis - 6 classes
- **pN** (Pathologic Node Stage): N0, N1, N2, N3 - 4 classes  
- **pM** (Pathologic Metastasis): M0, M1 - 2 classes
- **pCR** (Pathologic Complete Response): Complete, Not complete, DCIS only, LCIS only - 4 classes

---

## ✅ Tasks Completed

### 1. Environment Setup ✓
- Created virtual environment `mri_env`
- Installed all required dependencies:
  - PyTorch 2.9.0 + CUDA support
  - MONAI 1.5.0 (medical imaging)
  - timm, einops (vision transformers)
  - scikit-learn, pandas, seaborn, matplotlib (analysis)
  - pydicom, pynrrd, SimpleITK (medical image I/O)
  - openpyxl (Excel parsing)

### 2. Clinical Data Analysis ✓
- **Total patients**: 923 (924 rows with 1 header)
- **NAC patients**: ~292-312 patients received neoadjuvant chemotherapy
- **Key columns identified**:
  - Patient ID
  - Pathologic stage T (post-NAC): -1=TX, 0=T0, 1=T1, 2=T2, 3=T3, 4=T4, 5=Tis
  - Pathologic stage N (post-NAC): -1=NX, 0=N0, 1=N1, 2=N2, 3=N3
  - Pathologic stage M (post-NAC): -1=MX, 0=M0, 1=M1
  - Pathologic Response: 1=complete, 2=not complete, 3=DCIS only, 4=LCIS only, 5=unavailable
  - Near-complete Response: 0=not, 1=complete, 2=near-complete, 3=unavailable
  - Received NAC: 1=yes, 2=no

**Distribution (NAC patients only, ~292 patients)**:
- pT: T1 (124), T2 (61), TX (31), T0 (24), T3 (20)
- pN: N0 (110), N1 (64), NX (54), N2 (24), N3 (18)
- pM: MX (260), M0 (3), M1 (2)
- pCR: Not complete (224), Complete (64), Assessment unavailable (12), DCIS only (11)

### 3. Data Preprocessing Pipeline ✓
Created `preprocess_mri_data.py` that:
- ✅ Loads DICOM volumes from `Segmentation_Input_Automated_Download`
- ✅ Loads NRRD segmentation masks from `Segmentation_Outputs_293`
- ✅ Converts 3D volumes to 2D slices
- ✅ Normalizes each slice to [0, 1] range
- ✅ Saves as PNG images (8-bit RGB)
- ✅ Matches patients with clinical labels
- ✅ Filters for NAC patients only
- ✅ Generates metadata CSV with labels per slice

**Current Status**: Testing on 10 patients (RUNNING)

---

## 🔄 Next Steps

### 4. Custom Dataset Class (TODO)
Create `MRIClinicalDataset` PyTorch Dataset that:
- Loads preprocessed 2D slices
- Applies data augmentation (training only)
- Returns: images, optional masks, patient IDs, clinical labels
- Handles patient-level train/val/test splits (70/15/15)
- Groups slices by patient for volume-level aggregation

### 5. Clinical Prediction Head Architecture (TODO)
Design multi-task architecture:
```
Input: 2D MRI Slice (1024x1024 or 512x512)
   ↓
MRI-CORE Image Encoder (ViT-B, frozen initially)
   ↓ 
Feature Map: (B, 256, 64, 64)
   ↓
Global Average Pooling → (B, 256)
   ↓
┌──────────────┬──────────────┬──────────────┬──────────────┐
│   pT Head    │   pN Head    │   pM Head    │   pCR Head   │
│  (6 classes) │  (4 classes) │  (2 classes) │  (4 classes) │
│  FC(256→128) │  FC(256→128) │  FC(256→64)  │  FC(256→128) │
│  ReLU+Dropout│  ReLU+Dropout│  ReLU+Dropout│  ReLU+Dropout│
│  FC(128→6)   │  FC(128→4)   │  FC(64→2)    │  FC(128→4)   │
└──────────────┴──────────────┴──────────────┴──────────────┘
```

**Slice-to-Volume Aggregation Options**:
1. **Simple**: Average predictions across all slices
2. **Attention**: Learnable attention weights for each slice
3. **LSTM**: Sequential processing of slices
4. **Max Pooling**: Take max confidence prediction

### 6. Training Script (TODO)
Create `train_clinical_predictor.py`:
- Load MRI-CORE pre-trained weights from `mri_foundation.pth`
- Freeze encoder initially (feature extraction)
- Multi-task loss: Weighted CrossEntropy for each head
- Class weights to handle imbalanced labels
- Metrics: Accuracy, F1-score, AUC-ROC per task
- Early stopping on validation loss
- Checkpoint best model

### 7. Testing & Evaluation (TODO)
- Test on 5-10 patients first
- Evaluate preprocessing quality
- Check label distributions
- Verify model can load and forward pass
- Monitor GPU memory usage

### 8. Full Training (TODO)
- Train on all ~292 NAC patients
- Monitor convergence
- Analyze per-task performance
- Generate confusion matrices
- Clinical interpretation of results

---

## 💻 Computing Requirements Analysis

### Dataset Size Estimation

**Per Patient**:
- DICOM volume: ~150-200 slices
- Slice dimensions: 448x448 or 512x512
- After preprocessing: ~150 PNG images at ~50KB each = 7.5 MB/patient

**Total Dataset (~292 NAC patients)**:
- Images: 292 × 150 = ~43,800 slices
- Storage: 292 × 7.5 MB = ~2.2 GB preprocessed
- Raw DICOM: ~50-100 GB (but we process to PNG)

### Memory Requirements

**Training**:
- Batch size: 4-8 images
- Image size: 1024×1024×3 (RGB)
- MRI-CORE encoder: ViT-B (86M parameters)
- Prediction heads: ~2M parameters
- **Estimated GPU Memory**: 10-16 GB per GPU
  - With mixed precision (FP16): 6-10 GB

**Inference**:
- Batch size: 16-32
- **Estimated GPU Memory**: 4-8 GB

### Training Time Estimation

**Configuration**:
- ~43,800 slices, batch size 8 → 5,475 iterations/epoch
- Epochs: 50-100 (with early stopping)
- Single GPU (RTX 3090/A6000)

**Estimated Time**:
- Per epoch: ~10-15 minutes (ViT forward pass + backprop)
- Total: 8-25 hours for 50-100 epochs

### Hardware Recommendations

#### ✅ **LOCAL MACHINE (Your Setup)**
**Feasible if you have**:
- GPU: RTX 3080/3090 (10-24 GB VRAM) or RTX 4080/4090
- RAM: 32 GB+ system memory
- Storage: 50 GB free space (20 GB for data, 30 GB for cache/models)
- CPU: 8+ cores for data loading

**Training Strategy**:
- Freeze MRI-CORE encoder (feature extraction mode)
- Reduce batch size to 4
- Use mixed precision (FP16)
- Resize images to 512×512 if needed
- Enable gradient checkpointing

**Estimated Time**: 1-2 days

#### ⚠️ **CONSIDERATIONS FOR LOCAL**:
- **Pros**:
  - No cost
  - Full control
  - Easy debugging
  - No data transfer
  
- **Cons**:
  - Ties up your machine
  - Slower than multi-GPU
  - Risk of interruption

#### 🚀 **RECOMMENDED: GPU CLUSTER/CLOUD**
**If local is not feasible**:
- **Google Colab Pro+**: $50/month, A100 GPU (40GB), sufficient
- **AWS EC2**: g5.xlarge (NVIDIA A10G, 24GB), ~$1-2/hour
- **Lambda Labs**: RTX 4090/A6000, ~$0.50-1.00/hour
- **RunPod**: RTX 4090, ~$0.39/hour

**Estimated Cost**: $10-50 for full training

---

## 📊 Expected Results & Success Criteria

### Performance Targets
- **pCR Prediction (most important)**: AUC-ROC > 0.75
- **pT Stage**: Accuracy > 0.60 (6-class problem is hard)
- **pN Stage**: Accuracy > 0.65
- **pM Stage**: High imbalance, monitor carefully

### Clinical Significance
- **High performers**: pCR, pT stage (good response vs poor response)
- **Challenging**: pM (very few M1 cases)
- **Useful even with moderate performance**: Can aid treatment planning

---

## 🎯 Decision Point: Local vs Cloud

### Run Locally If:
- ✅ You have RTX 3080/3090/4090 (≥10 GB VRAM)
- ✅ Can dedicate machine for 1-2 days
- ✅ Want to iterate and debug easily
- ✅ No budget for cloud

### Use Cloud/Cluster If:
- ⚠️ GPU has < 10 GB VRAM
- ⚠️ Need results quickly (< 12 hours)
- ⚠️ Want to run multiple experiments in parallel
- ⚠️ Don't want to tie up local machine

---

## 📝 Files Generated So Far

1. `analyze_clinical_data.py` - Clinical data exploration
2. `parse_clinical_data.py` - Multi-header Excel parsing
3. `clinical_data_processed.csv` - Cleaned clinical data
4. `patient_outcomes_mapping.csv` - Patient outcomes
5. `find_nac_patients.py` - NAC patient identifier
6. `debug_nac_column.py` - Data format debugger
7. `preprocess_mri_data.py` - **Main preprocessing pipeline**
8. `preprocessed_data/` - Output directory (being generated)
   - `images/` - PNG slices
   - `masks/` - Segmentation masks
   - `metadata.csv` - Slice-level labels
   - `preprocessing_summary.json` - Statistics

---

## 🔮 Next Session Plan

1. ✅ Check preprocessing results (should be done by now)
2. 📝 Create `MRIClinicalDataset` class
3. 📝 Create multi-task prediction architecture
4. 📝 Create training script
5. 🧪 Test on 5 patients
6. ❓ **DECISION**: Local vs Cloud based on your GPU specs
7. 🚀 Run full training
8. 📊 Evaluate and analyze results

**Estimated time to working model**: 4-8 hours of implementation + training time

---

*Last Updated: Current Session*
*Status: Preprocessing in progress, architecture design ready*
