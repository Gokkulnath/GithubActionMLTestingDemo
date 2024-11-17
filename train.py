import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model.network import MNISTNet
from datetime import datetime
import os

def train_model():
    # Force CPU
    device = torch.device('cpu')
    
    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load MNIST dataset
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Initialize model
    model = MNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'Batch: {batch_idx}, Loss: {loss.item():.4f}, Accuracy: {100.*correct/total:.2f}%')
    
    final_accuracy = 100. * correct / total
    print(f'Final Training Accuracy: {final_accuracy:.2f}%')
    
    # Save model with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f'models/mnist_model_{timestamp}.pth'
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), save_path)
    
    return final_accuracy, save_path

if __name__ == '__main__':
    train_model() 