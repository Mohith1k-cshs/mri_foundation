#!/bin/bash
#SBATCH --job-name=mri_preprocess
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=cpu

# MRI Preprocessing Job - CPU-bound task
# Processes ~292 NAC patients, converts DICOM to PNG
# Estimated time: 2-4 hours

echo "=========================================="
echo "MRI Preprocessing Job Started"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Load required modules (adjust for your HPC)
module purge
module load python/3.11
module load cuda/12.4  # For PyTorch CUDA support

# Activate virtual environment
source mri_env/bin/activate

# Print environment info
echo ""
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"
echo ""

# Create output directories
mkdir -p preprocessed_data
mkdir -p logs

# Run preprocessing
echo "Starting preprocessing of all NAC patients..."
python preprocess_mri_data.py

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Preprocessing completed successfully!"
    echo "End time: $(date)"
    echo "=========================================="
    
    # Print statistics
    echo ""
    echo "Output statistics:"
    echo "  Metadata entries: $(wc -l < preprocessed_data/metadata.csv)"
    echo "  Image files: $(find preprocessed_data/images -name "*.png" | wc -l)"
    echo "  Disk usage: $(du -sh preprocessed_data)"
    
    # Trigger training job (optional - remove if you want to check preprocessing first)
    # echo ""
    # echo "Submitting training job..."
    # sbatch train_job.sh
else
    echo ""
    echo "=========================================="
    echo "✗ Preprocessing failed with exit code $?"
    echo "Check error log: logs/preprocess_${SLURM_JOB_ID}.err"
    echo "=========================================="
    exit 1
fi
