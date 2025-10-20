"""
Custom PyTorch Dataset for MRI Clinical Prediction

Loads preprocessed 2D MRI slices with clinical outcome labels.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

class MRIClinicalDataset(Dataset):
    """
    Dataset for MRI slices with clinical outcome labels
    
    Args:
        metadata_csv: Path to metadata.csv from preprocessing
        data_root: Root directory containing images/ and masks/
        split: 'train', 'val', or 'test'
        image_size: Target image size (default: 1024)
        use_masks: Whether to load segmentation masks
        transform: Additional transforms to apply
        augment: Whether to apply data augmentation (train only)
    """
    
    def __init__(self, 
                 metadata_csv,
                 data_root,
                 split='train',
                 image_size=1024,
                 use_masks=False,
                 transform=None,
                 augment=True,
                 random_seed=42):
        
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.use_masks = use_masks
        self.transform = transform
        self.augment = augment and (split == 'train')
        
        # Load metadata
        self.metadata = pd.read_csv(metadata_csv)
        
        # Split by patient (not by slice) to avoid data leakage
        self.patient_ids = self.metadata['patient_id'].unique()
        
        # Split patients: 70% train, 15% val, 15% test
        train_patients, temp_patients = train_test_split(
            self.patient_ids, test_size=0.3, random_state=random_seed
        )
        val_patients, test_patients = train_test_split(
            temp_patients, test_size=0.5, random_state=random_seed
        )
        
        # Filter metadata for current split
        if split == 'train':
            self.data = self.metadata[self.metadata['patient_id'].isin(train_patients)]
        elif split == 'val':
            self.data = self.metadata[self.metadata['patient_id'].isin(val_patients)]
        elif split == 'test':
            self.data = self.metadata[self.metadata['patient_id'].isin(test_patients)]
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
        
        self.data = self.data.reset_index(drop=True)
        
        # Define transforms
        self.setup_transforms()
        
        print(f"[{split.upper()}] Loaded {len(self.data)} slices from {len(self.data['patient_id'].unique())} patients")
        
        # Print label distributions
        self.print_label_distribution()
    
    def setup_transforms(self):
        """Setup image transforms"""
        base_transforms = [
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])  # SAM/ImageNet normalization
        ]
        
        if self.augment:
            # Add augmentations for training
            augmentation_transforms = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            ] + base_transforms
            self.img_transform = transforms.Compose(augmentation_transforms)
        else:
            self.img_transform = transforms.Compose(base_transforms)
        
        # Mask transform (no normalization, just resize)
        self.mask_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), 
                            interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
    
    def print_label_distribution(self):
        """Print distribution of labels in this split"""
        print(f"  Label distributions:")
        for label in ['pT', 'pN', 'pM', 'pCR']:
            if label in self.data.columns:
                dist = self.data[label].value_counts().sort_index()
                print(f"    {label}: {dict(dist)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get one slice with labels"""
        row = self.data.iloc[idx]
        
        # Load image
        img_path = self.data_root / row['image_path']
        image = Image.open(img_path).convert('RGB')
        image = self.img_transform(image)
        
        # Load mask if requested
        mask = None
        if self.use_masks and pd.notna(row['mask_path']):
            mask_path = self.data_root / row['mask_path']
            mask = Image.open(mask_path).convert('L')
            mask = self.mask_transform(mask)
        
        # Extract labels
        labels = {}
        
        # pT: Pathologic T stage (-1 to 5)
        if 'pT' in row and pd.notna(row['pT']):
            pt_value = str(row['pT']).strip()
            try:
                labels['pT'] = int(pt_value) + 1  # Shift to 0-6 range (TX=-1 becomes 0)
            except:
                labels['pT'] = 0  # Default to TX (unknown)
        else:
            labels['pT'] = 0
        
        # pN: Pathologic N stage (-1 to 3)
        if 'pN' in row and pd.notna(row['pN']):
            pn_value = str(row['pN']).strip()
            try:
                labels['pN'] = int(pn_value) + 1  # Shift to 0-4 range
            except:
                labels['pN'] = 0
        else:
            labels['pN'] = 0
        
        # pM: Pathologic M stage (-1 to 1)
        if 'pM' in row and pd.notna(row['pM']):
            pm_value = str(row['pM']).strip()
            try:
                labels['pM'] = int(pm_value) + 1  # Shift to 0-2 range
            except:
                labels['pM'] = 0
        else:
            labels['pM'] = 0
        
        # pCR: Pathologic complete response (1-5)
        if 'pCR' in row and pd.notna(row['pCR']):
            pcr_value = str(row['pCR']).strip()
            try:
                # Map: 1=complete, 2=not complete, 3=DCIS, 4=LCIS, 5=unavailable
                labels['pCR'] = int(pcr_value) - 1  # Shift to 0-4 range
            except:
                labels['pCR'] = 4  # unavailable
        else:
            labels['pCR'] = 4
        
        # Convert labels to tensors
        labels_tensor = {
            'pT': torch.tensor(labels['pT'], dtype=torch.long),
            'pN': torch.tensor(labels['pN'], dtype=torch.long),
            'pM': torch.tensor(labels['pM'], dtype=torch.long),
            'pCR': torch.tensor(labels['pCR'], dtype=torch.long),
        }
        
        # Return
        output = {
            'image': image,
            'patient_id': row['patient_id'],
            'slice_idx': row['slice_idx'],
            **labels_tensor
        }
        
        if mask is not None:
            output['mask'] = mask
        
        return output


def get_class_weights(dataset, task):
    """Calculate class weights for imbalanced data"""
    # Count occurrences of each class
    labels = []
    for i in range(len(dataset)):
        sample = dataset.data.iloc[i]
        label_col = {'pT': 'pT', 'pN': 'pN', 'pM': 'pM', 'pCR': 'pCR'}[task]
        
        if label_col in sample and pd.notna(sample[label_col]):
            try:
                if task == 'pCR':
                    label_value = int(str(sample[label_col]).strip()) - 1
                else:
                    label_value = int(str(sample[label_col]).strip()) + 1
                labels.append(label_value)
            except:
                pass
    
    # Calculate weights (inverse frequency)
    class_counts = np.bincount(labels)
    total = len(labels)
    weights = total / (len(class_counts) * class_counts + 1e-6)
    
    return torch.FloatTensor(weights)


# Test the dataset
if __name__ == "__main__":
    # Test loading
    metadata_csv = "preprocessed_data/metadata.csv"
    data_root = "preprocessed_data"
    
    print("="*80)
    print("Testing MRIClinicalDataset")
    print("="*80)
    
    # Create datasets
    train_dataset = MRIClinicalDataset(
        metadata_csv=metadata_csv,
        data_root=data_root,
        split='train',
        image_size=512,  # Use smaller size for testing
        use_masks=True,
        augment=True
    )
    
    val_dataset = MRIClinicalDataset(
        metadata_csv=metadata_csv,
        data_root=data_root,
        split='val',
        image_size=512,
        use_masks=True,
        augment=False
    )
    
    test_dataset = MRIClinicalDataset(
        metadata_csv=metadata_csv,
        data_root=data_root,
        split='test',
        image_size=512,
        use_masks=True,
        augment=False
    )
    
    # Test loading one sample
    print("\n" + "="*80)
    print("Testing sample loading")
    print("="*80)
    sample = train_dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Mask shape: {sample['mask'].shape if 'mask' in sample else 'No mask'}")
    print(f"Patient ID: {sample['patient_id']}")
    print(f"Slice index: {sample['slice_idx']}")
    print(f"Labels:")
    for key in ['pT', 'pN', 'pM', 'pCR']:
        print(f"  {key}: {sample[key].item()}")
    
    # Calculate class weights
    print("\n" + "="*80)
    print("Class Weights (for handling imbalance)")
    print("="*80)
    for task in ['pT', 'pN', 'pM', 'pCR']:
        weights = get_class_weights(train_dataset, task)
        print(f"{task}: {weights}")
    
    print("\n✓ Dataset test complete!")
