"""
Training Script for MRI-CORE Clinical Predictor

Trains multi-task model to predict TNM staging from pre-treatment breast MRI
"""

import os
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import pandas as pd

# Import custom modules
import cfg as mri_cfg
from dataset_clinical import MRIClinicalDataset, get_class_weights
from model_clinical import create_model


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss with task-specific weights and class weights
    """
    def __init__(self, tasks, class_weights_dict, task_weights=None):
        super().__init__()
        self.tasks = tasks
        self.task_weights = task_weights if task_weights else {t: 1.0 for t in tasks}
        
        # Create loss functions with class weights
        self.criterions = {}
        for task in tasks:
            if task in class_weights_dict and class_weights_dict[task] is not None:
                weight = class_weights_dict[task].cuda() if torch.cuda.is_available() else class_weights_dict[task]
                self.criterions[task] = nn.CrossEntropyLoss(weight=weight)
            else:
                self.criterions[task] = nn.CrossEntropyLoss()
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: dict of {task: logits}
            targets: dict of {task: labels}
        
        Returns:
            total_loss, loss_dict
        """
        losses = {}
        total_loss = 0.0
        
        for task in self.tasks:
            if task in predictions and task in targets:
                task_loss = self.criterions[task](predictions[task], targets[task])
                losses[task] = task_loss
                total_loss += self.task_weights[task] * task_loss
        
        return total_loss, losses


def evaluate(model, dataloader, criterion, device):
    """Evaluate model on validation/test set"""
    model.eval()
    
    all_predictions = {task: [] for task in ['pT', 'pN', 'pM', 'pCR']}
    all_targets = {task: [] for task in ['pT', 'pN', 'pM', 'pCR']}
    total_loss = 0.0
    task_losses = {task: 0.0 for task in ['pT', 'pN', 'pM', 'pCR']}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            images = batch['image'].to(device)
            targets = {task: batch[task].to(device) for task in ['pT', 'pN', 'pM', 'pCR']}
            
            # Forward
            predictions = model(images)
            
            # Loss
            loss, losses = criterion(predictions, targets)
            total_loss += loss.item()
            for task, task_loss in losses.items():
                task_losses[task] += task_loss.item()
            
            # Collect predictions and targets
            for task in ['pT', 'pN', 'pM', 'pCR']:
                pred_labels = predictions[task].argmax(dim=1).cpu().numpy()
                true_labels = targets[task].cpu().numpy()
                all_predictions[task].extend(pred_labels)
                all_targets[task].extend(true_labels)
    
    # Calculate metrics
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_task_losses = {task: loss / num_batches for task, loss in task_losses.items()}
    
    metrics = {'loss': avg_loss, 'task_losses': avg_task_losses}
    
    for task in ['pT', 'pN', 'pM', 'pCR']:
        y_true = np.array(all_targets[task])
        y_pred = np.array(all_predictions[task])
        
        # Filter out invalid labels (if any)
        valid_mask = (y_true >= 0) & (y_true < len(np.unique(y_true)))
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]
        
        if len(y_true) > 0:
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            metrics[f'{task}_accuracy'] = acc
            metrics[f'{task}_f1'] = f1
        else:
            metrics[f'{task}_accuracy'] = 0.0
            metrics[f'{task}_f1'] = 0.0
    
    return metrics, all_predictions, all_targets


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0.0
    task_losses = {task: 0.0 for task in ['pT', 'pN', 'pM', 'pCR']}
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        targets = {task: batch[task].to(device) for task in ['pT', 'pN', 'pM', 'pCR']}
        
        # Forward
        predictions = model(images)
        
        # Loss
        loss, losses = criterion(predictions, targets)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track losses
        total_loss += loss.item()
        for task, task_loss in losses.items():
            task_losses[task] += task_loss.item()
        
        # Update progress bar
        if batch_idx % 10 == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'pT': f'{losses["pT"].item():.4f}',
                'pCR': f'{losses["pCR"].item():.4f}'
            })
    
    # Average losses
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_task_losses = {task: loss / num_batches for task, loss in task_losses.items()}
    
    return avg_loss, avg_task_losses


def plot_training_curves(train_history, val_history, save_dir):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Training Progress')
    
    # Overall loss
    axes[0, 0].plot(train_history['loss'], label='Train')
    axes[0, 0].plot(val_history['loss'], label='Val')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Task-specific metrics
    tasks = ['pT', 'pN', 'pM', 'pCR']
    for idx, task in enumerate(tasks):
        row = (idx + 1) // 3
        col = (idx + 1) % 3
        
        if f'{task}_accuracy' in val_history:
            axes[row, col].plot(val_history[f'{task}_accuracy'], label='Accuracy')
            axes[row, col].plot(val_history[f'{task}_f1'], label='F1')
            axes[row, col].set_title(f'{task} Metrics')
            axes[row, col].set_xlabel('Epoch')
            axes[row, col].set_ylabel('Score')
            axes[row, col].legend()
            axes[row, col].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_curves.png', dpi=150)
    plt.close()


def save_confusion_matrices(all_predictions, all_targets, save_dir):
    """Save confusion matrices for each task"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle('Confusion Matrices (Validation Set)')
    
    tasks = ['pT', 'pN', 'pM', 'pCR']
    for idx, task in enumerate(tasks):
        row = idx // 2
        col = idx % 2
        
        y_true = np.array(all_targets[task])
        y_pred = np.array(all_predictions[task])
        
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[row, col], cmap='Blues')
        axes[row, col].set_title(f'{task}')
        axes[row, col].set_xlabel('Predicted')
        axes[row, col].set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'confusion_matrices.png', dpi=150)
    plt.close()


def main(args):
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = MRIClinicalDataset(
        metadata_csv=args.metadata_csv,
        data_root=args.data_root,
        split='train',
        image_size=args.image_size,
        use_masks=args.use_masks,
        augment=True
    )
    
    val_dataset = MRIClinicalDataset(
        metadata_csv=args.metadata_csv,
        data_root=args.data_root,
        split='val',
        image_size=args.image_size,
        use_masks=args.use_masks,
        augment=False
    )
    
    test_dataset = MRIClinicalDataset(
        metadata_csv=args.metadata_csv,
        data_root=args.data_root,
        split='test',
        image_size=args.image_size,
        use_masks=args.use_masks,
        augment=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Calculate class weights
    print("\nCalculating class weights...")
    class_weights = {}
    for task in ['pT', 'pN', 'pM', 'pCR']:
        weights = get_class_weights(train_dataset, task)
        if len(weights) > 0:
            class_weights[task] = weights
            print(f"  {task} weights: {weights}")
        else:
            class_weights[task] = None
    
    # Create model
    print("\nCreating model...")
    mri_args = mri_cfg.parse_args()
    mri_args.image_size = args.image_size
    mri_args.if_encoder_adapter = False
    mri_args.if_mask_decoder_adapter = False
    
    model = create_model(
        args=mri_args,
        mri_core_checkpoint=args.mri_core_checkpoint,
        freeze_encoder=args.freeze_encoder
    )
    model = model.to(device)
    
    # Loss and optimizer
    criterion = MultiTaskLoss(
        tasks=['pT', 'pN', 'pM', 'pCR'],
        class_weights_dict=class_weights,
        task_weights={'pT': 1.0, 'pN': 1.0, 'pM': 0.5, 'pCR': 2.0}  # pCR is most important
    )
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Freeze encoder: {args.freeze_encoder}")
    
    train_history = {'loss': [], 'task_losses': {task: [] for task in ['pT', 'pN', 'pM', 'pCR']}}
    val_history = {'loss': [], 'task_losses': {task: [] for task in ['pT', 'pN', 'pM', 'pCR']}}
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*80}")
        
        # Train
        train_loss, train_task_losses = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        train_history['loss'].append(train_loss)
        for task, loss in train_task_losses.items():
            if task not in train_history['task_losses']:
                train_history['task_losses'][task] = []
            train_history['task_losses'][task].append(loss)
        
        # Validate
        val_metrics, val_predictions, val_targets = evaluate(model, val_loader, criterion, device)
        val_history['loss'].append(val_metrics['loss'])
        
        for task in ['pT', 'pN', 'pM', 'pCR']:
            if f'{task}_accuracy' not in val_history:
                val_history[f'{task}_accuracy'] = []
                val_history[f'{task}_f1'] = []
            val_history[f'{task}_accuracy'].append(val_metrics[f'{task}_accuracy'])
            val_history[f'{task}_f1'].append(val_metrics[f'{task}_f1'])
        
        # Print summary
        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print("\nTask-specific metrics:")
        for task in ['pT', 'pN', 'pM', 'pCR']:
            print(f"  {task}: Acc={val_metrics[f'{task}_accuracy']:.4f}, F1={val_metrics[f'{task}_f1']:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_metrics['loss'])
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_epoch = epoch
            patience_counter = 0
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_metrics': val_metrics,
            }
            torch.save(checkpoint, output_dir / 'best_model.pth')
            print(f"\n✓ Saved best model (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.early_stop_patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break
        
        # Plot progress
        if epoch % 5 == 0:
            plot_training_curves(train_history, val_history, output_dir)
    
    # Final evaluation on test set
    print(f"\n{'='*80}")
    print("Final Evaluation on Test Set")
    print(f"{'='*80}")
    
    # Load best model
    checkpoint = torch.load(output_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics, test_predictions, test_targets = evaluate(model, test_loader, criterion, device)
    
    print(f"\nTest Loss: {test_metrics['loss']:.4f}")
    print("\nTest Metrics:")
    for task in ['pT', 'pN', 'pM', 'pCR']:
        print(f"  {task}: Acc={test_metrics[f'{task}_accuracy']:.4f}, F1={test_metrics[f'{task}_f1']:.4f}")
    
    # Save confusion matrices
    save_confusion_matrices(test_predictions, test_targets, output_dir)
    
    # Save results
    results = {
        'best_epoch': best_epoch,
        'val_metrics': checkpoint['val_metrics'],
        'test_metrics': test_metrics,
        'train_history': {k: [float(v) if isinstance(v, (int, float)) else v for v in vals] 
                          for k, vals in train_history.items() if isinstance(vals, list)},
        'val_history': {k: [float(v) if isinstance(v, (int, float)) else v for v in vals] 
                        for k, vals in val_history.items() if isinstance(vals, list)},
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Training complete! Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MRI-CORE Clinical Predictor")
    
    # Data args
    parser.add_argument('--metadata_csv', type=str, default='preprocessed_data/metadata.csv')
    parser.add_argument('--data_root', type=str, default='preprocessed_data')
    parser.add_argument('--mri_core_checkpoint', type=str, default='mri_foundation.pth')
    
    # Model args
    parser.add_argument('--image_size', type=int, default=512, help='Input image size (512 or 1024)')
    parser.add_argument('--freeze_encoder', action='store_true', default=True)
    parser.add_argument('--use_masks', action='store_true', default=False)
    
    # Training args
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--early_stop_patience', type=int, default=15)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Output
    parser.add_argument('--output_dir', type=str, default='outputs/clinical_predictor')
    
    args = parser.parse_args()
    
    # Print configuration
    print("="*80)
    print("MRI-CORE Clinical Predictor - Training")
    print("="*80)
    print("\nConfiguration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("="*80)
    
    main(args)
