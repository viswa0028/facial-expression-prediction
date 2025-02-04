import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.functional as F
class Convolution(nn.Module):
    def __init__(self, num_classes):
        super(Convolution, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        # Activation layer
        self.relu = nn.ReLU()  # Define ReLU once and reuse it
        
        # Pooling layer
        self.poolinglayer = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=18432, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Pass through convolutional layers with ReLU and pooling
        x = self.relu(self.conv1(x))
        x = self.poolinglayer(self.relu(self.conv2(x)))
        x = self.poolinglayer(self.relu(self.conv3(x)))
        
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        
        # Pass through fully connected layers with ReLU and dropout
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.Resize((48, 48)),                  # Resize to 48x48
    transforms.ToTensor(),                        # Convert image to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])   # Normalize pixel values to [-1, 1]
])

train_dataset = datasets.ImageFolder(root="TRAIN DATASET PATH", transform=transform)
transform1 = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48,48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5],std = [0.5])
])
test_dataset  = datasets.ImageFolder(root="TEST DATASET PATH",transform=transform1)
from torch.utils.data import DataLoader

# Define batch size
batch_size = 32

# Create DataLoaders for training and testing
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Convolution(7).to(device)
criterian = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr = 0.001)
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images,labels in train_loader:
        images ,labels = images.to(device),labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterian(output,labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")
torch.save(model.state_dict(), "facial_expression_cnn.pth")
print("Model saved as facial_expression_cnn.pth")
