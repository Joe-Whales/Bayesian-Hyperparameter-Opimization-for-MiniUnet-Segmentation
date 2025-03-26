import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.multiprocessing import set_start_method

# Try to set the start method for multiprocessing
try:
    set_start_method('spawn', force=True)
except RuntimeError:
    # Method already set, ignore
    pass

class SquareSegmentationDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # Get image and mask paths
        self.image_dir = os.path.join(data_dir, split, 'images')
        self.mask_dir = os.path.join(data_dir, split, 'masks')
        
        # Ensure directories exist
        if not os.path.exists(self.image_dir) or not os.path.exists(self.mask_dir):
            raise ValueError(f"Directories not found: {self.image_dir} or {self.mask_dir}")
        
        # Get filenames and ensure corresponding pairs exist
        image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.png')]
        self.image_paths = []
        self.mask_paths = []
        
        for img_file in sorted(image_files):
            img_path = os.path.join(self.image_dir, img_file)
            mask_path = os.path.join(self.mask_dir, img_file)  # Assuming same filename
            
            if os.path.exists(mask_path):
                self.image_paths.append(img_path)
                self.mask_paths.append(mask_path)
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No valid image-mask pairs found in {data_dir}/{split}")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image and mask
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        if image is None or mask is None:
            raise ValueError(f"Failed to load image or mask at index {idx}: {self.image_paths[idx]}")
        
        # Normalize image and convert mask to binary
        image = image.astype(np.float32) / 255.0
        mask = (mask > 128).astype(np.float32)  # Convert to binary mask
        
        # Convert to tensor
        image = torch.from_numpy(image).float().unsqueeze(0)  # Add channel dimension
        mask = torch.from_numpy(mask).float().unsqueeze(0)    # Add channel dimension
        
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            
        return image, mask

def get_datasets(data_dir):
    """
    Get the train, validation, and test datasets.
    
    Args:
        data_dir: Path to the dataset directory
        
    Returns:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
    """
    
    train_dataset = SquareSegmentationDataset(data_dir, split='train')
    val_dataset = SquareSegmentationDataset(data_dir, split='val')
    test_dataset = SquareSegmentationDataset(data_dir, split='test')
    
    return train_dataset, val_dataset, test_dataset

def get_dataloaders(data_dir, batch_size=8, num_workers=0, persistent_workers=False, prefetch_factor=2):
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        data_dir: Directory containing the dataset splits
        batch_size: Batch size for training and evaluation
        num_workers: Number of worker processes for data loading
        persistent_workers: Whether to keep worker processes alive between iterations
    
    Returns:
        train_loader, val_loader, test_loader: DataLoader objects
    """
    # Create datasets
    try:
        train_dataset = SquareSegmentationDataset(data_dir, split='train')
        val_dataset = SquareSegmentationDataset(data_dir, split='val')
        test_dataset = SquareSegmentationDataset(data_dir, split='test')
    except Exception as e:
        print(f"Error creating datasets: {e}")
        raise
    
    # Configure DataLoader options
    dataloader_kwargs = {
        'batch_size': batch_size,
        'pin_memory': torch.cuda.is_available(),
        'num_workers': num_workers,
        'prefetch_factor': prefetch_factor,
    }
    
    # Add persistent_workers only if num_workers > 0
    if num_workers > 0:
        dataloader_kwargs['persistent_workers'] = persistent_workers
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        shuffle=True, 
        **dataloader_kwargs
    )
    
    val_loader = DataLoader(
        val_dataset, 
        shuffle=False, 
        **dataloader_kwargs
    )
    
    test_loader = DataLoader(
        test_dataset, 
        shuffle=False, 
        **dataloader_kwargs
    )
    
    print(f"Created data loaders with {len(train_dataset)} training, "
          f"{len(val_dataset)} validation, and {len(test_dataset)} test samples")
    
    return train_loader, val_loader, test_loader