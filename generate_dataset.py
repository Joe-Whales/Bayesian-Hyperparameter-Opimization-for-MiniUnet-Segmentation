import os
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.draw import polygon, ellipse
import math
import random
import yaml

def create_directories(base_path):
    """Create the necessary directories for the dataset."""
    # Create main directory if it doesn't exist
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    # Create subdirectories
    for split in ['train', 'val', 'test']:
        for subdir in ['images', 'masks']:
            path = os.path.join(base_path, split, subdir)
            if not os.path.exists(path):
                os.makedirs(path)
    
    print(f"Created directory structure in {base_path}")

def generate_square_mask(img_size, min_size=0.1, max_size=0.5, rotation_range=(-45, 45)):
    """Generate a mask with a rotated square."""
    mask = np.zeros((img_size, img_size), dtype=np.uint8)
    
    # Determine square size (as a fraction of image size)
    size_factor = np.random.uniform(min_size, max_size)
    square_size = int(img_size * size_factor)
    
    # Determine square position
    max_pos = img_size - square_size
    if max_pos <= 0:
        max_pos = 1  # Prevent potential issues with very large squares
    pos_x = np.random.randint(0, max_pos)
    pos_y = np.random.randint(0, max_pos)
    
    # Create square coordinates (centered at origin)
    half_size = square_size // 2
    x1, y1 = -half_size, -half_size
    x2, y2 = half_size, -half_size
    x3, y3 = half_size, half_size
    x4, y4 = -half_size, half_size
    
    # Random rotation
    angle = np.random.uniform(*rotation_range)
    angle_rad = math.radians(angle)
    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)
    
    # Rotate square coordinates
    def rotate_point(x, y):
        return (x * cos_angle - y * sin_angle, x * sin_angle + y * cos_angle)
    
    rx1, ry1 = rotate_point(x1, y1)
    rx2, ry2 = rotate_point(x2, y2)
    rx3, ry3 = rotate_point(x3, y3)
    rx4, ry4 = rotate_point(x4, y4)
    
    # Translate to final position
    center_x = pos_x + half_size
    center_y = pos_y + half_size
    
    poly_x = np.array([rx1 + center_x, rx2 + center_x, rx3 + center_x, rx4 + center_x], dtype=np.int32)
    poly_y = np.array([ry1 + center_y, ry2 + center_y, ry3 + center_y, ry4 + center_y], dtype=np.int32)
    
    # Draw filled polygon
    rr, cc = polygon(poly_y, poly_x, mask.shape)
    mask[rr, cc] = 255
    
    return mask

def add_random_blobs(mask, num_blobs, min_size, max_size, min_stretch, max_stretch, min_alpha, max_alpha):
    """Add random blob-like shapes to the image with varying transparency."""
    # Create a copy of the mask for blending
    result = mask.copy().astype(np.float32)
    blob_mask = np.zeros_like(mask, dtype=np.uint8)  # For tracking blob locations
    
    for _ in range(num_blobs):
        # Create a blank image for the blob
        blob = np.zeros_like(mask, dtype=np.float32)
        
        # Random blob position
        center_y = np.random.randint(0, mask.shape[0])
        center_x = np.random.randint(0, mask.shape[1])
        
        # Random blob size
        radius = np.random.uniform(min_size, max_size) * mask.shape[0] / 2
        
        # Random stretch factor (1.0 = circle, >1.0 = ellipse)
        stretch = np.random.uniform(min_stretch, max_stretch)
        
        # Random rotation
        angle = np.random.uniform(0, 360)
        
        # Calculate semi-axes
        a = radius
        b = radius * stretch
        
        # Draw ellipse
        rr, cc = ellipse(center_y, center_x, a, b, mask.shape, rotation=np.radians(angle))
        blob[rr, cc] = 255
        
        # Random transparency/alpha
        alpha = np.random.uniform(min_alpha, max_alpha)
        
        # Add blob to the mask with transparency
        result = np.where(blob > 0, 
                          result * (1 - alpha) + blob * alpha, 
                          result)
        
        # Track blob locations for ground truth
        blob_mask[rr, cc] = 255
    
    # Convert back to uint8
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result, blob_mask

def add_noise_and_blur(image, noise_level=0.2, blur_kernel_range=(3, 15)):
    """Add noise and blur to an image."""
    # Add Gaussian noise
    noise = np.random.normal(0, noise_level * 255, image.shape).astype(np.int32)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)
    
    # Apply random blur
    kernel_size = np.random.randint(*blur_kernel_range) // 2 * 2 + 1  # Ensure odd kernel size
    image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    return image

def generate_dataset(config):
    """Generate the full dataset with configuration from YAML."""
    # Extract parameters from config
    base_path = config['dataset']['path']
    img_size = config['dataset']['image_size']
    num_images = config['dataset']['num_images']
    train_ratio = config['dataset']['train_ratio']
    val_ratio = config['dataset']['val_ratio']
    
    # Square generation parameters
    min_size = config['square']['min_size']
    max_size = config['square']['max_size']
    rotation_range = (
        config['square']['min_rotation'],
        config['square']['max_rotation']
    )
    
    # Blob parameters
    use_blobs = config.get('blobs', {}).get('enabled', False)
    if use_blobs:
        num_blobs = config['blobs']['num_blobs']
        min_blob_size = config['blobs']['min_size']
        max_blob_size = config['blobs']['max_size']
        min_stretch = config['blobs']['min_stretch']
        max_stretch = config['blobs']['max_stretch']
        min_alpha = config['blobs']['min_alpha']
        max_alpha = config['blobs']['max_alpha']
    
    # Image augmentation parameters
    noise_level = config['augmentation']['noise_level']
    blur_kernel_range = (
        config['augmentation']['min_blur_kernel'],
        config['augmentation']['max_blur_kernel']
    )
    
    # Create directories
    create_directories(base_path)
    
    # Calculate split sizes
    num_train = int(num_images * train_ratio)
    num_val = int(num_images * val_ratio)
    num_test = num_images - num_train - num_val
    
    # Generate data for each split
    splits = {
        'train': num_train,
        'val': num_val,
        'test': num_test
    }
    
    # Set random seed for reproducibility
    seed = config.get('random_seed', 42)
    np.random.seed(seed)
    random.seed(seed)
    
    # Save the configuration to the dataset directory
    with open(os.path.join(base_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    # Generate images for each split
    for split, count in splits.items():
        print(f"Generating {count} images for {split} set...")
        for i in tqdm(range(count)):
            # Generate mask with a rotated square
            mask = generate_square_mask(
                img_size, 
                min_size=min_size, 
                max_size=max_size,
                rotation_range=rotation_range
            )
            
            # Create a copy of the mask for the input image
            image = mask.copy()
            
            # Add random blobs if enabled
            if use_blobs:
                image, blob_mask = add_random_blobs(
                    image,
                    num_blobs=num_blobs,
                    min_size=min_blob_size,
                    max_size=max_blob_size,
                    min_stretch=min_stretch,
                    max_stretch=max_stretch,
                    min_alpha=min_alpha,
                    max_alpha=max_alpha
                )
            
            # Add noise and blur to create final input image
            image = add_noise_and_blur(
                image,
                noise_level=noise_level,
                blur_kernel_range=blur_kernel_range
            )
            
            # Save files
            image_filename = os.path.join(base_path, split, 'images', f"{split}_{i:04d}.png")
            mask_filename = os.path.join(base_path, split, 'masks', f"{split}_{i:04d}.png")
            
            cv2.imwrite(image_filename, image)
            cv2.imwrite(mask_filename, mask)
    
    print(f"Generated {num_images} images in total.")
    
    # Visualize a few examples
    visualize_examples(base_path, num_examples=config.get('visualization', {}).get('num_examples', 3))

def visualize_examples(base_path, num_examples=3):
    """Visualize a few examples from the training set."""
    train_images_dir = os.path.join(base_path, 'train', 'images')
    train_masks_dir = os.path.join(base_path, 'train', 'masks')
    
    image_files = os.listdir(train_images_dir)
    np.random.shuffle(image_files)
    
    plt.figure(figsize=(12, 4 * num_examples))
    
    for i in range(min(num_examples, len(image_files))):
        filename = image_files[i]
        
        image_path = os.path.join(train_images_dir, filename)
        mask_path = os.path.join(train_masks_dir, filename)
        
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        plt.subplot(num_examples, 3, i*3 + 1)
        plt.imshow(image, cmap='gray')
        plt.title(f"Input Image {i+1}")
        plt.axis('off')
        
        plt.subplot(num_examples, 3, i*3 + 2)
        plt.imshow(mask, cmap='gray')
        plt.title(f"Ground Truth {i+1}")
        plt.axis('off')
        
        plt.subplot(num_examples, 3, i*3 + 3)
        overlay = np.zeros((*image.shape, 3), dtype=np.uint8)
        overlay[..., 0] = image
        overlay[..., 1] = image
        overlay[..., 2] = image
        overlay[mask > 128, 0] = 0  # Set red channel to 0
        overlay[mask > 128, 1] = 255  # Set green channel to max
        overlay[mask > 128, 2] = 0  # Set blue channel to 0
        plt.imshow(overlay)
        plt.title(f"Overlay {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, 'examples.png'))
    plt.close()
    print(f"Example visualization saved to {os.path.join(base_path, 'examples.png')}")

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate synthetic square segmentation dataset')
    parser.add_argument('--config', type=str, default='config.yaml', 
                        help='Path to configuration YAML file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Generate the dataset
    generate_dataset(config)
    
    print("Dataset generation complete!")