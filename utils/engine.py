import torch
import time

def train_one_epoch(model, criterion, optimizer, scheduler, train_loader, device, args):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    # Start the timer for this epoch
    epoch_start_time = time.time()
    total_step = len(train_loader)
    for i, (images, labels) in enumerate(train_loader):
        # Move images and labels to device
        images = images.to(device)
        labels = labels.to(device)
        # print(images.size())
        # Forward pass
        outputs = model(images)
        
        if args.model == 'SwiftFormer_L3':
            outputs = outputs[0]
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Apply the warm-up learning rate scheduler
        if scheduler:
            scheduler.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        if args.cutmix:
            _, labels = torch.max(labels.data, 1)
            
        correct += (predicted == labels).sum().item()
        print(f"\rStep [{i+1}/{total_step}], training Loss: {train_loss/len(train_loader):.4f}, training Aaccuracy: {((correct / total) * 100.0):.2f}, Time: {time.time()-epoch_start_time:.2f}", end='')
    # Calculate training loss and accuracy
    train_loss /= len(train_loader)
    train_accuracy = (correct / total) * 100.0
    epoch_time = time.time()-epoch_start_time
    return train_loss, train_accuracy, epoch_time


def evalute(model, criterion, valid_loader, device, args):
    start = time.time()
    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in valid_loader:
            # Move images and labels to device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            # Calculate accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    # Calculate validation loss and accuracy
    valid_loss /= len(valid_loader)
    valid_accuracy = (correct / total) * 100.0
    valid_time = time.time()-start
    
    return valid_loss, valid_accuracy, valid_time
    
    
    
    
    
