import torch
from utils import binary_accuracy, calculate_metrics

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    
    for _, (images, masks) in enumerate(dataloader):
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        
        outputs = torch.sigmoid(outputs)
        loss = criterion(outputs, masks)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        running_acc += binary_accuracy(outputs, masks) * images.size(0)
        
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_acc / len(dataloader.dataset)
    
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    all_metrics = {
        'f1_score': 0.0,
        'iou_score': 0.0
    }
    
    with torch.no_grad():
        for _, (images, masks) in enumerate(dataloader):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            
            outputs = torch.sigmoid(outputs)
            loss = criterion(outputs, masks)
            
            running_loss += loss.item() * images.size(0)
            running_acc += binary_accuracy(outputs, masks) * images.size(0)
            
            batch_metrics = calculate_metrics(outputs, masks)
            for k, v in batch_metrics.items():
                all_metrics[k] += v * images.size(0)
        
    metrics = {'loss': running_loss / len(dataloader.dataset),'accuracy': running_acc / len(dataloader.dataset)}
    
    for k, v in all_metrics.items():
        metrics[k] = v / len(dataloader.dataset)
    
    return metrics

def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler=None, num_epochs=50, patience=10, device='cpu', verbose=True):
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
        'val_iou': []
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    counter = 0
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        if scheduler is not None:
            scheduler.step(val_metrics['loss'])
        
        if ((epoch + 1) % 5 == 0 or epoch == num_epochs - 1) and verbose:
            print(f"Epoch [{epoch+1}/{num_epochs}] - Train loss: {train_loss:.4f} - Train acc: {train_acc:.4f} - Val loss: {val_metrics['loss']:.4f} - Val acc: {val_metrics['accuracy']:.4f} - Val F1: {val_metrics['f1_score']:.4f} - Val IoU: {val_metrics['iou_score']:.4f}")
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1_score'])
        history['val_iou'].append(val_metrics['iou_score'])
        
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_model_state = model.state_dict().copy()
            counter = 0
        else:
            counter += 1
        
        if counter >= patience:
            if verbose:
                print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history

