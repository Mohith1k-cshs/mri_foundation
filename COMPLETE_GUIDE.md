# 🏥 MRI-CORE Clinical Prediction - Complete Implementation Guide

## 📋 Executive Summary

**What we've built**: A complete pipeline to predict post-chemotherapy pathological outcomes from pre-treatment breast MRI using the MRI-CORE foundation model.

**Status**: ✅ Fully implemented and tested on 10 patients. Ready to scale to all 292 NAC patients.

**Clinical Goals**:
- Predict **pT** (tumor stage post-NAC)
- Predict **pN** (nodal stage post-NAC)
- Predict **pM** (metastasis status)
- Predict **pCR** (pathological complete response) - **PRIMARY OUTCOME**

---

## 🗂️ Project Structure

```
mri_foundation/
├── mri_foundation.pth                 # Pre-trained MRI-CORE model (86M params)
├── Clinical_and_Other_Features.xlsx   # Clinical outcomes (923 patients)
├── requirements.txt                   # Dependencies
│
├── IMPLEMENTATION FILES (Created)
├── preprocess_mri_data.py             # ✅ DICOM → PNG preprocessing
├── dataset_clinical.py                # ✅ PyTorch Dataset for MRI slices
├── model_clinical.py                  # ✅ MRI-CORE + prediction heads
├── train_clinical_predictor.py        # ✅ Complete training script
│
├── UTILITY FILES
├── analyze_clinical_data.py           # Data exploration
├── parse_clinical_data.py             # Excel parsing
├── find_nac_patients.py               # NAC patient finder
├── debug_nac_column.py                # Data debugging
│
├── GENERATED DATA
├── patient_outcomes_mapping.csv       # Patient → outcomes mapping
├── preprocessed_data/                 # Preprocessed MRI slices
│   ├── images/                        # PNG images (1,646 slices from 10 patients)
│   ├── masks/                         # Segmentation masks
│   ├── metadata.csv                   # Slice-level metadata
│   └── preprocessing_summary.json     # Statistics
│
└── outputs/                           # Training outputs (will be created)
    └── clinical_predictor/
        ├── best_model.pth             # Best checkpoint
        ├── results.json               # Metrics
        ├── training_curves.png        # Loss/accuracy plots
        └── confusion_matrices.png     # Per-task confusion matrices
```

---

## 🚀 Step-by-Step Execution Guide

### **Step 1: Preprocess ALL NAC Patients** ⏳

Currently only 10 patients are preprocessed. Process all ~292 NAC patients:

```powershell
# Full preprocessing (will take ~2-4 hours)
C:/Users/kanthamnenm/Projects/mri_foundation/mri_env/Scripts/python.exe preprocess_mri_data.py

# OR process in batches
C:/Users/kanthamnenm/Projects/mri_foundation/mri_env/Scripts/python.exe preprocess_mri_data.py --max_patients 50
C:/Users/kanthamnenm/Projects/mri_foundation/mri_env/Scripts/python.exe preprocess_mri_data.py --max_patients 100
# ... continue
```

**Expected Output**:
- ~43,800 slices (150 slices × 292 patients)
- ~2.2 GB storage
- Updated `preprocessed_data/metadata.csv` with all slices

**⚠️ IMPORTANT**: This step is I/O intensive but CPU-only. You can run it overnight or while working.

---

### **Step 2: Test the Training Pipeline** 🧪

Test on current 10 patients first (quick validation):

```powershell
C:/Users/kanthamnenm/Projects/mri_foundation/mri_env/Scripts/python.exe train_clinical_predictor.py `
    --metadata_csv preprocessed_data/metadata.csv `
    --data_root preprocessed_data `
    --mri_core_checkpoint mri_foundation.pth `
    --image_size 512 `
    --batch_size 4 `
    --epochs 10 `
    --freeze_encoder `
    --output_dir outputs/test_run
```

**Time**: ~10-15 minutes
**Purpose**: Verify everything works before full training

---

### **Step 3: Full Training** 🎯

#### **Option A: Local Training (If you have GPU ≥10GB VRAM)**

```powershell
# For RTX 3080/3090/4090 or similar
C:/Users/kanthamnenm/Projects/mri_foundation/mri_env/Scripts/python.exe train_clinical_predictor.py `
    --metadata_csv preprocessed_data/metadata.csv `
    --data_root preprocessed_data `
    --mri_core_checkpoint mri_foundation.pth `
    --image_size 512 `
    --batch_size 8 `
    --epochs 100 `
    --learning_rate 1e-4 `
    --freeze_encoder `
    --early_stop_patience 15 `
    --output_dir outputs/clinical_predictor
```

**Time**: 12-24 hours
**Requirements**:
- GPU: ≥10 GB VRAM
- RAM: 32 GB
- Storage: 20 GB free

**If GPU memory is limited**, reduce batch size and image size:
```powershell
--batch_size 4 `
--image_size 384
```

#### **Option B: Cloud/Cluster Training**

If local GPU insufficient, use Google Colab Pro+ or cloud GPU:

1. **Google Colab Pro+ ($50/month)**:
   - Upload preprocessed data to Google Drive
   - Use A100 GPU (40GB VRAM)
   - Run training notebook

2. **RunPod (~$0.39/hour)**:
   ```bash
   # SSH into RunPod instance
   git clone <your_repo>
   cd mri_foundation
   pip install -r requirements.txt
   python train_clinical_predictor.py <args>
   ```

3. **AWS/Lambda Labs**: Similar setup

**Cost**: $10-50 for full training

---

### **Step 4: Monitor Training** 📊

Training will display:
```
Epoch 15/100
Train Loss: 0.8234
Val Loss: 0.9156

Task-specific metrics:
  pT: Acc=0.6234, F1=0.5987
  pN: Acc=0.7012, F1=0.6845
  pM: Acc=0.9500, F1=0.8234 (mostly MX, high accuracy expected)
  pCR: Acc=0.7456, F1=0.7123 ← MOST IMPORTANT

✓ Saved best model (val_loss=0.9156)
```

**Key Metrics to Watch**:
1. **pCR Accuracy & F1**: Primary outcome, target >0.70
2. **pT Accuracy**: Target >0.60 (6-class problem is hard)
3. **Validation Loss**: Should decrease steadily
4. **Early Stopping**: Will trigger if no improvement for 15 epochs

**Outputs Generated**:
- `outputs/clinical_predictor/best_model.pth` - Best checkpoint
- `outputs/clinical_predictor/training_curves.png` - Loss/accuracy plots
- `outputs/clinical_predictor/confusion_matrices.png` - Per-task confusion
- `outputs/clinical_predictor/results.json` - All metrics

---

### **Step 5: Evaluate Results** 📈

After training completes, review:

1. **Test Set Performance** (printed at end):
   ```
   Test Loss: 0.9234
   Test Metrics:
     pT: Acc=0.6123, F1=0.5934
     pN: Acc=0.6892, F1=0.6734
     pM: Acc=0.9401, F1=0.8123
     pCR: Acc=0.7345, F1=0.7012 ← SUCCESS if >0.70!
   ```

2. **Confusion Matrices**: Check `confusion_matrices.png`
   - Are errors random or systematic?
   - Which classes are confused (e.g., T1 vs T2)?

3. **Training Curves**: Check `training_curves.png`
   - Is model converging?
   - Any overfitting (train >> val)?
   - Did early stopping trigger correctly?

---

## 💻 Hardware Requirements Decision Tree

```
Do you have a GPU?
├─ NO → Use Google Colab Pro+ ($50/month, easiest)
└─ YES
   │
   ├─ Is VRAM ≥ 10 GB? (RTX 3080+, RTX 4080+, A6000, etc.)
   │  └─ YES → ✅ Train locally!
   │          Set: --batch_size 8 --image_size 512
   │          Time: 12-24 hours
   │
   └─ Is VRAM < 10 GB? (RTX 3060, RTX 2080, etc.)
      │
      ├─ Is VRAM ≥ 8 GB?
      │  └─ YES → Try locally with reduced settings
      │          Set: --batch_size 4 --image_size 384
      │          May work, ~15-30 hours
      │
      └─ VRAM < 8 GB
         └─ Use Cloud GPU (RunPod $0.39/hour = ~$10 total)
```

---

## 📊 Expected Results & Clinical Interpretation

### **Performance Targets**

| Task | Metric | Target | Clinical Significance |
|------|--------|--------|----------------------|
| **pCR** | Acc / F1 | >0.70 / >0.70 | **Primary outcome**: Identifies complete responders |
| **pT** | Acc / F1 | >0.60 / >0.55 | Tumor stage: Predicts residual tumor burden |
| **pN** | Acc / F1 | >0.65 / >0.60 | Nodal involvement: Guides surgical planning |
| **pM** | Acc / F1 | >0.90 / >0.80 | Metastasis: High imbalance (mostly MX) |

### **Clinical Impact**

**If pCR accuracy > 0.75**:
- ✅ **Excellent**: Model can identify complete responders
- 💡 Use for: Treatment escalation/de-escalation decisions
- 💡 Clinical trial potential: Personalized NAC regimens

**If pCR accuracy 0.70-0.75**:
- ✅ **Good**: Useful for clinical decision support
- 💡 Use for: Second opinion, risk stratification
- 💡 Combine with clinical factors for better prediction

**If pCR accuracy 0.65-0.70**:
- ⚠️ **Moderate**: Research value, not ready for clinic
- 💡 Next steps: Try unfreezing encoder, more data, radiomics features

**If pCR accuracy < 0.65**:
- ⚠️ **Below target**: MRI may not have sufficient predictive signal
- 💡 Consider: Adding clinical features (age, grade, ER/PR/HER2 status)

---

## 🔧 Troubleshooting Guide

### **Issue: CUDA Out of Memory**

**Solution 1**: Reduce batch size
```powershell
--batch_size 4  # or even 2
```

**Solution 2**: Reduce image size
```powershell
--image_size 384  # or 256
```

**Solution 3**: Enable mixed precision (add to script if needed)
```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

### **Issue: Training Very Slow (>1 hour/epoch)**

**Causes**:
1. Image size too large → use 512 instead of 1024
2. Too many workers → reduce `--num_workers`
3. Slow I/O → move data to SSD

**Solution**:
```powershell
--image_size 512 --num_workers 2
```

### **Issue: Model Not Converging**

**Check**:
1. Learning rate too high/low → try 5e-5 or 5e-4
2. Class imbalance not handled → weights are calculated automatically
3. Frozen encoder limiting capacity → try `--no-freeze_encoder`

### **Issue: Overfitting (Train loss << Val loss)**

**Solutions**:
1. More data augmentation (already enabled)
2. Higher dropout (edit `model_clinical.py` dropout=0.5)
3. More regularization: `--weight_decay 0.05`
4. Early stopping will handle this automatically

---

## 📝 Next Steps After Training

### **Immediate Analysis**

1. **Review Confusion Matrices**:
   - Which classes are hardest to distinguish?
   - Any systematic errors?

2. **Feature Importance**:
   - Which slices contribute most? (future: attention visualization)

3. **Error Analysis**:
   - Review misclassified cases
   - Any imaging artifacts causing failures?

### **Model Improvements**

1. **Unfreeze Encoder** (if results < target):
   ```powershell
   python train_clinical_predictor.py --no-freeze_encoder --learning_rate 1e-5
   ```

2. **Add Clinical Features**:
   - Combine imaging with age, tumor grade, biomarkers

3. **Ensemble Methods**:
   - Train multiple models with different seeds
   - Average predictions

### **Clinical Translation**

1. **External Validation**:
   - Test on different institution's data

2. **Prospective Study**:
   - Evaluate in real clinical workflow

3. **Explainability**:
   - Generate attention maps
   - Identify which breast regions drive predictions

---

## 📖 Key Files Documentation

### **preprocess_mri_data.py**
- Converts DICOM volumes to 2D PNG slices
- Normalizes to [0, 1] range
- Matches with clinical labels
- Filters for NAC patients only

### **dataset_clinical.py**
- PyTorch Dataset for loading slices
- Handles train/val/test splits by patient
- Applies data augmentation
- Calculates class weights for imbalanced data

### **model_clinical.py**
- Loads MRI-CORE encoder (ViT-B, 86M params)
- Adds multi-task prediction heads
- Supports frozen or fine-tuned encoder

### **train_clinical_predictor.py**
- Complete training loop
- Multi-task loss with class weights
- Learning rate scheduling
- Early stopping
- Automatic evaluation and plotting

---

## 🎓 Understanding the Architecture

```
Pre-treatment MRI Slice (1024×1024 or 512×512)
         ↓
┌────────────────────────────────────┐
│   MRI-CORE Image Encoder (ViT-B)  │ ← Frozen initially
│   - 86M parameters                 │
│   - Pretrained on diverse MRI      │
└────────────────────────────────────┘
         ↓
    Feature Map (B, 256, H/16, W/16)
         ↓
    Global Average Pooling
         ↓
    Features (B, 256)
         ↓
┌───────────────┬───────────────┬───────────────┬───────────────┐
│   pT Head     │   pN Head     │   pM Head     │   pCR Head    │
│               │               │               │               │
│ FC(256→128)   │ FC(256→128)   │ FC(256→64)    │ FC(256→128)   │
│ ReLU+Dropout  │ ReLU+Dropout  │ ReLU+Dropout  │ ReLU+Dropout  │
│ FC(128→64)    │ FC(128→64)    │ FC(64→2)      │ FC(128→64)    │
│ ReLU+Dropout  │ ReLU+Dropout  │               │ ReLU+Dropout  │
│ FC(64→7)      │ FC(64→5)      │               │ FC(64→5)      │
│               │               │               │               │
│ 7 classes     │ 5 classes     │ 3 classes     │ 5 classes     │
│ (TX-T4, Tis)  │ (NX-N3)       │ (MX, M0, M1)  │ (Complete,    │
│               │               │               │  Not, DCIS,   │
│               │               │               │  LCIS, N/A)   │
└───────────────┴───────────────┴───────────────┴───────────────┘
         ↓            ↓               ↓               ↓
    Predictions for post-NAC pathological outcomes
```

**Key Design Choices**:
1. **Frozen Encoder**: Faster training, less overfitting, works well with small datasets
2. **Multi-Task Learning**: Shares representations, improves generalization
3. **Class Weights**: Handles imbalanced labels (e.g., few M1 cases)
4. **Task Weights**: pCR weighted 2×, most clinically important

---

## 🎯 Success Criteria

### **Technical Success**
- [x] ✅ Preprocessing pipeline works
- [x] ✅ Dataset loading works
- [x] ✅ Model forward pass works
- [x] ✅ Training loop works
- [ ] ⏳ Full training completes without errors
- [ ] ⏳ Model achieves pCR accuracy > 0.70

### **Clinical Success**
- [ ] ⏳ Model predictions correlate with pathology
- [ ] ⏳ Identifies complete responders (pCR=1)
- [ ] ⏳ Outperforms clinical factors alone
- [ ] ⏳ Generalizes to test set

---

## 📞 Quick Reference Commands

```powershell
# 1. Preprocess all data
python preprocess_mri_data.py

# 2. Test training (quick)
python train_clinical_predictor.py --epochs 10 --output_dir outputs/test

# 3. Full training (local GPU)
python train_clinical_predictor.py --batch_size 8 --epochs 100

# 4. Full training (limited GPU)
python train_clinical_predictor.py --batch_size 4 --image_size 384 --epochs 100

# 5. Check dataset
python dataset_clinical.py

# 6. Check model
python model_clinical.py
```

---

## 📚 Citation & Acknowledgments

**MRI-CORE Paper**:
```bibtex
@article{dong2024mricore,
  title={MRI-CORE: A Foundation Model for Magnetic Resonance Imaging},
  author={Dong, Haoyu and Chen, Yuwen and Gu, Hanxue and Konz, Nicholas and Chen, Yaqian and Li, Qihang and Mazurowski, Maciej A},
  journal={arXiv preprint arXiv:2404.09957},
  year={2024}
}
```

---

## ✅ Final Checklist Before Full Training

- [x] ✅ Environment set up
- [x] ✅ Dependencies installed
- [x] ✅ Clinical data parsed
- [x] ✅ Preprocessing tested on 10 patients
- [x] ✅ Dataset class works
- [x] ✅ Model architecture validated
- [x] ✅ Training script ready
- [ ] ⏳ All ~292 NAC patients preprocessed
- [ ] ⏳ GPU ready (local or cloud)
- [ ] ⏳ Storage space confirmed (~20 GB)
- [ ] ⏳ Run full training
- [ ] ⏳ Analyze results
- [ ] ⏳ Clinical interpretation

---

**You are ready to go! 🚀**

**Estimated Time to Results**:
- Preprocessing: 2-4 hours (can run overnight)
- Training: 12-24 hours (local) or 6-12 hours (cloud A100)
- **Total**: 1-2 days

**Next Action**: Run full preprocessing on all NAC patients, then start training!

