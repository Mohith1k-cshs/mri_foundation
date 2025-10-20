#!/bin/bash
#SBATCH --job-name=mri_train_multi
#SBATCH --output=logs/train_multi_%j.out
#SBATCH --error=logs/train_multi_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:4
#SBATCH --partition=gpu

# Multi-GPU Training Job - For faster training with DGX Spark
# Uses 4x A100 GPUs with Distributed Data Parallel
# Estimated time: 2-3 hours

echo "=========================================="
echo "Multi-GPU Training Started"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: 4x A100"
echo "Start time: $(date)"
echo "=========================================="

# Load modules
module purge
module load python/3.11
module load cuda/12.4
module load cudnn/8.9
module load nccl/2.18  # For multi-GPU communication

# Activate environment
source mri_env/bin/activate

# GPU info
echo ""
echo "GPU Information:"
nvidia-smi
echo ""

# Check preprocessing
if [ ! -f "preprocessed_data/metadata.csv" ]; then
    echo "ERROR: Preprocessing not complete!"
    exit 1
fi

# Multi-GPU training with torchrun
echo "Starting 4-GPU distributed training..."
torchrun --nproc_per_node=4 \
    train_clinical_predictor.py \
    --metadata_csv preprocessed_data/metadata.csv \
    --data_root preprocessed_data \
    --mri_core_checkpoint mri_foundation.pth \
    --image_size 512 \
    --batch_size 64 \
    --epochs 100 \
    --learning_rate 0.0001 \
    --freeze_encoder \
    --output_dir outputs/clinical_predictor

echo ""
echo "Training complete: $(date)"
