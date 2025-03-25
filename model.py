import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from tqdm.notebook import tqdm
from sklearn.metrics import f1_score, jaccard_score
import yaml
import random
from PIL import Image

# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load configuration
def load_config(config_path="square_segmentation_dataset/config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Load dataset configuration
dataset_config = load_config()
print("Dataset configuration loaded!")
# Define dataset class
class SquareSegmentationDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # Get image and mask paths
        self.image_dir = os.path.join(data_dir, split, 'images')
        self.mask_dir = os.path.join(data_dir, split, 'masks')
        
        self.image_paths = sorted([os.path.join(self.image_dir, img) 
                                  for img in os.listdir(self.image_dir)
                                  if img.endswith('.png')])
        
        self.mask_paths = sorted([os.path.join(self.mask_dir, mask) 
                                 for mask in os.listdir(self.mask_dir)
                                 if mask.endswith('.png')])
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image and mask
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        # Normalize image and mask
        image = image / 255.0
        mask = (mask > 128).astype(np.float32)  # Convert to binary mask
        
        # Convert to tensor
        image = torch.from_numpy(image).float().unsqueeze(0)  # Add channel dimension
        mask = torch.from_numpy(mask).float().unsqueeze(0)    # Add channel dimension
        
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            
        return image, mask

# Create dataloaders
def get_dataloaders(data_dir, batch_size=8, num_workers=2):
    train_dataset = SquareSegmentationDataset(data_dir, split='train')
    val_dataset = SquareSegmentationDataset(data_dir, split='val')
    test_dataset = SquareSegmentationDataset(data_dir, split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader

# Initialize dataloaders
data_dir = dataset_config['dataset']['path']
train_loader, val_loader, test_loader = get_dataloaders(data_dir)

# Display dataset sizes
print(f"Train samples: {len(train_loader.dataset)}")
print(f"Validation samples: {len(val_loader.dataset)}")
print(f"Test samples: {len(test_loader.dataset)}")

# Visualize some samples
def visualize_samples(dataloader, num_samples=3):
    # Get a batch of samples
    images, masks = next(iter(dataloader))
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    for i in range(min(num_samples, len(images))):
        # Get image and mask
        image = images[i, 0].numpy()
        mask = masks[i, 0].numpy()
        
        # Display image
        axes[i, 0].imshow(image, cmap='gray')
        axes[i, 0].set_title(f"Input Image {i+1}")
        axes[i, 0].axis('off')
        
        # Display mask
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title(f"Ground Truth {i+1}")
        axes[i, 1].axis('off')
        
        # Display overlay
        overlay = np.zeros((image.shape[0], image.shape[1], 3))
        overlay[..., 0] = image
        overlay[..., 1] = image
        overlay[..., 2] = image
        overlay[mask > 0.5, 0] = 0
        overlay[mask > 0.5, 1] = 1
        overlay[mask > 0.5, 2] = 0
        
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title(f"Overlay {i+1}")
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

# Visualize training samples
print("Training samples:")
visualize_samples(train_loader)
