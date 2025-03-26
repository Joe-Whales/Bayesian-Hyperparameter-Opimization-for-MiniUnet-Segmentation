import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Create dataloaders
def get_dataloaders(data_dir, batch_size=8, num_workers=0):  # Setting num_workers=0 for Windows compatibility
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
