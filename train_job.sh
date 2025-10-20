#!/bin/bash
#SBATCH --job-name=mri_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu

# MRI Clinical Predictor Training Job - GPU training
# Trains multi-task model for TNM staging prediction
# Estimated time: 4-6 hours on A100

echo "=========================================="
echo "MRI Clinical Predictor Training Started"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Load required modules (adjust for your HPC)
module purge
module load python/3.11
module load cuda/12.4
module load cudnn/8.9

# Activate virtual environment
source mri_env/bin/activate

# Print environment info
echo ""
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"
echo ""

# GPU info
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo ""

# Verify CUDA is available
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}')"
echo ""

# Check if preprocessing completed
if [ ! -f "preprocessed_data/metadata.csv" ]; then
    echo "ERROR: Preprocessing not complete. Run preprocess_job.sh first!"
    exit 1
fi

TOTAL_SLICES=$(wc -l < preprocessed_data/metadata.csv)
echo "Found preprocessed data: $TOTAL_SLICES slices"
echo ""

# Create output directory
mkdir -p outputs/clinical_predictor
mkdir -p logs

# Training configuration
BATCH_SIZE=32           # Adjust based on GPU memory (32 for A100)
IMAGE_SIZE=512          # Full resolution
EPOCHS=100              # With early stopping
LEARNING_RATE=0.0001    # Standard for fine-tuning
FREEZE_ENCODER=true     # Freeze MRI-CORE encoder

echo "Training Configuration:"
echo "  Batch size: $BATCH_SIZE"
echo "  Image size: $IMAGE_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Learning rate: $LEARNING_RATE"
echo "  Freeze encoder: $FREEZE_ENCODER"
echo ""

# Run training
echo "Starting training..."
python train_clinical_predictor.py \
    --metadata_csv preprocessed_data/metadata.csv \
    --data_root preprocessed_data \
    --mri_core_checkpoint mri_foundation.pth \
    --image_size $IMAGE_SIZE \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --learning_rate $LEARNING_RATE \
    --freeze_encoder \
    --output_dir outputs/clinical_predictor

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Training completed successfully!"
    echo "End time: $(date)"
    echo "=========================================="
    
    # Print results summary
    echo ""
    echo "Results saved to: outputs/clinical_predictor/"
    echo "  - best_model.pth (trained model)"
    echo "  - results.json (metrics)"
    echo "  - training_curves.png (loss/accuracy plots)"
    echo "  - confusion_matrices.png (per-task analysis)"
    
    # Display final metrics
    if [ -f "outputs/clinical_predictor/results.json" ]; then
        echo ""
        echo "Final Test Metrics:"
        python -c "import json; results = json.load(open('outputs/clinical_predictor/results.json')); print('\n'.join([f'  {k}: {v:.4f}' for k, v in results['test_metrics'].items()]))" 2>/dev/null || echo "  (See results.json for details)"
    fi
else
    echo ""
    echo "=========================================="
    echo "✗ Training failed with exit code $?"
    echo "Check error log: logs/train_${SLURM_JOB_ID}.err"
    echo "=========================================="
    exit 1
fi

echo ""
echo "Next steps:"
echo "1. Review results: cat outputs/clinical_predictor/results.json"
echo "2. View plots: outputs/clinical_predictor/*.png"
echo "3. Analyze confusion matrices for clinical interpretation"
