"""
MRI-CORE Clinical Predictor Model

Multi-task prediction of TNM staging from pre-treatment breast MRI
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.sam import sam_model_registry
import cfg

class ClinicalPredictionHead(nn.Module):
    """
    Multi-task prediction head for clinical outcomes
    """
    def __init__(self, 
                 feature_dim=256, 
                 num_classes_dict={'pT': 7, 'pN': 5, 'pM': 3, 'pCR': 5},
                 dropout=0.3):
        super().__init__()
        
        self.num_classes_dict = num_classes_dict
        
        # Separate prediction heads for each task
        self.heads = nn.ModuleDict()
        
        for task, num_classes in num_classes_dict.items():
            if task == 'pM':
                # pM has fewer classes, use smaller head
                self.heads[task] = nn.Sequential(
                    nn.Linear(feature_dim, 64),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(64, num_classes)
                )
            else:
                # Standard head for other tasks
                self.heads[task] = nn.Sequential(
                    nn.Linear(feature_dim, 128),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(64, num_classes)
                )
    
    def forward(self, features):
        """
        Args:
            features: (B, feature_dim) - aggregated features
        
        Returns:
            dict of predictions for each task
        """
        predictions = {}
        for task, head in self.heads.items():
            predictions[task] = head(features)
        return predictions


class MRICORExClinicalPredictor(nn.Module):
    """
    Complete model: MRI-CORE encoder + Clinical prediction heads
    """
    def __init__(self, 
                 args,
                 mri_core_checkpoint,
                 num_classes_dict={'pT': 7, 'pN': 5, 'pM': 3, 'pCR': 5},
                 freeze_encoder=True,
                 feature_aggregation='gap'):  # 'gap', 'attention', or 'lstm'
        super().__init__()
        
        self.freeze_encoder = freeze_encoder
        self.feature_aggregation = feature_aggregation
        
        # Load MRI-CORE encoder
        print(f"Loading MRI-CORE from {mri_core_checkpoint}...")
        self.mri_core = sam_model_registry['vit_b'](
            args,
            checkpoint=mri_core_checkpoint,
            num_classes=1,
            image_size=args.image_size,
            pretrained_sam=False
        )
        
        # Freeze encoder if specified
        if freeze_encoder:
            print("Freezing MRI-CORE encoder...")
            for param in self.mri_core.image_encoder.parameters():
                param.requires_grad = False
        
        # Feature dimension from MRI-CORE encoder neck output
        # Neck outputs (B, 256, H/16, W/16) - we'll pool to get (B, 256)
        self.feature_dim = 256
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Optional: Attention-based aggregation
        if feature_aggregation == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(self.feature_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 1)
            )
        
        # Prediction heads
        self.prediction_head = ClinicalPredictionHead(
            feature_dim=self.feature_dim,
            num_classes_dict=num_classes_dict,
            dropout=0.3
        )
    
    def extract_features(self, images):
        """
        Extract features from MRI-CORE encoder
        
        Args:
            images: (B, 3, H, W)
        
        Returns:
            features: (B, 256)
        """
        # Pass through MRI-CORE encoder
        with torch.set_grad_enabled(not self.freeze_encoder):
            encoder_output = self.mri_core.image_encoder(images)
            # encoder_output shape: (B, 256, H/16, W/16)
        
        # Global average pooling
        features = self.gap(encoder_output)  # (B, 256, 1, 1)
        features = features.view(features.size(0), -1)  # (B, 256)
        
        return features
    
    def forward(self, images):
        """
        Forward pass
        
        Args:
            images: (B, 3, H, W) - batch of MRI slices
        
        Returns:
            predictions: dict of predictions for each task
                - pT: (B, 7)
                - pN: (B, 5)
                - pM: (B, 3)
                - pCR: (B, 5)
        """
        # Extract features
        features = self.extract_features(images)
        
        # Predict outcomes
        predictions = self.prediction_head(features)
        
        return predictions


def create_model(args, mri_core_checkpoint, freeze_encoder=True):
    """Helper function to create the model"""
    model = MRICORExClinicalPredictor(
        args=args,
        mri_core_checkpoint=mri_core_checkpoint,
        freeze_encoder=freeze_encoder
    )
    return model


# Test the model
if __name__ == "__main__":
    import cfg
    
    print("="*80)
    print("Testing MRICORExClinicalPredictor")
    print("="*80)
    
    # Parse args
    args = cfg.parse_args()
    args.image_size = 512  # Use smaller size for testing
    args.if_encoder_adapter = False
    args.if_mask_decoder_adapter = False
    
    # Create model
    model = create_model(
        args=args,
        mri_core_checkpoint="mri_foundation.pth",
        freeze_encoder=True
    )
    
    # Test forward pass
    batch_size = 4
    dummy_images = torch.randn(batch_size, 3, 512, 512)
    
    print(f"\nInput shape: {dummy_images.shape}")
    
    with torch.no_grad():
        predictions = model(dummy_images)
    
    print("\nOutput shapes:")
    for task, pred in predictions.items():
        print(f"  {task}: {pred.shape} (logits for {pred.shape[1]} classes)")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    
    print("\n✓ Model test complete!")
