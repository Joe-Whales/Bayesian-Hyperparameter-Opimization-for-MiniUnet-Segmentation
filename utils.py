import matplotlib.pyplot as plt
import numpy as np
from model import MiniUNet
import torch
import random
from sklearn.metrics import f1_score, jaccard_score

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

# Initialize model
def get_model(model_config, device='cpu'):
    model = MiniUNet(
        n_channels=1,  # Grayscale input
        n_classes=1,   # Binary segmentation (square vs background)
        base_filters=model_config.get('base_filters', 16),
        bilinear=model_config.get('bilinear', True),
        depth=model_config.get('depth', 3)
    )
    return model.to(device)

# Test the model with a random input
def test_model(model, input_size=(1, 1, 128, 128), device='cpu'):
    x = torch.randn(input_size).to(device)
    with torch.no_grad():
        output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    return output.shape

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def binary_accuracy(pred, target, threshold=0.5):
    """Calculate binary accuracy."""
    pred_binary = (pred > threshold).float()
    correct = (pred_binary == target).float().sum()
    acc = correct / target.numel()
    return acc.item()

def calculate_metrics(pred, target, threshold=0.5):
    """Calculate segmentation metrics."""
    pred_np = (pred.squeeze() > threshold).cpu().numpy().flatten()
    target_np = target.squeeze().cpu().numpy().flatten()
    
    # Calculate F1 score and IoU
    f1 = f1_score(target_np, pred_np, zero_division=1)
    iou = jaccard_score(target_np, pred_np, zero_division=1)
    
    return {
        'f1_score': f1,
        'iou_score': iou
    }

def plot_history(history):
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot metrics
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.plot(history['val_f1'], label='Validation F1')
    ax2.plot(history['val_iou'], label='Validation IoU')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Metric Value')
    ax2.set_title('Training and Validation Metrics')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def visualize_predictions(model, dataloader, num_samples=3, device='cpu'):
    # Get a batch of samples
    images, masks = next(iter(dataloader))
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        preds = torch.sigmoid(model(images.to(device)))
    
    # Convert predictions to numpy
    preds = preds.cpu().numpy()
    
    # Create figure
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    
    for i in range(min(num_samples, len(images))):
        # Get image, mask and prediction
        image = images[i, 0].numpy()
        mask = masks[i, 0].numpy()
        pred = preds[i, 0]
        
        # Display image
        axes[i, 0].imshow(image, cmap='gray')
        axes[i, 0].set_title(f"Input Image {i+1}")
        axes[i, 0].axis('off')
        
        # Display ground truth
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title(f"Ground Truth {i+1}")
        axes[i, 1].axis('off')
        
        # Display prediction
        axes[i, 2].imshow(pred, cmap='gray')
        axes[i, 2].set_title(f"Prediction {i+1}")
        axes[i, 2].axis('off')
        
        # Display overlay
        pred_binary = (pred > 0.5).astype(np.float32)
        overlay = np.zeros((image.shape[0], image.shape[1], 3))
        overlay[..., 0] = image
        overlay[..., 1] = image
        overlay[..., 2] = image
        
        # Highlight true positives in green
        tp = (pred_binary == 1) & (mask == 1)
        overlay[tp, 0] = 0
        overlay[tp, 1] = 1
        overlay[tp, 2] = 0
        
        # Highlight false positives in red
        fp = (pred_binary == 1) & (mask == 0)
        overlay[fp, 0] = 1
        overlay[fp, 1] = 0
        overlay[fp, 2] = 0
        
        # Highlight false negatives in blue
        fn = (pred_binary == 0) & (mask == 1)
        overlay[fn, 0] = 0
        overlay[fn, 1] = 0
        overlay[fn, 2] = 1
        
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title(f"Overlay {i+1}")
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.show()

def save_model(model, model_config, train_config, history, path="mini_unet_model.pth"):
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_config': model_config,
        'train_config': train_config,
        'history': history
    }
    torch.save(save_dict, path)
    print(f"Model saved to {path}")

def load_model(path="mini_unet_model.pth"):
    save_dict = torch.load(path)
    model = get_model(save_dict['model_config'])
    model.load_state_dict(save_dict['model_state_dict'])
    return model, save_dict['model_config'], save_dict['train_config'], save_dict['history']