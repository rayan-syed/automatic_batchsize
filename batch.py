import os
import sys
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import datetime

# Dataset Class
class CustomDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image

# Neural network from another project
class SimpleCNN(nn.Module):
    def __init__(self, num_conv_layers=60, input_channels=3):
        super(SimpleCNN, self).__init__()
        self.layers = num_conv_layers
        
        # Define the convolutional layers dynamically
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(in_channels=input_channels if i == 0 else 16, out_channels=16, kernel_size=3, stride=1, padding=1)
            for i in range(num_conv_layers)
        ])
        
        self.fc1 = nn.Linear(16 * 256 * 256, 10)  

    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            x = torch.relu(x)
        
        x = x.view(x.size(0), -1)  
        x = self.fc1(x)
        return x

def train(dataloader, model, criterion, optimizer, epochs):
    # Training loop
    for epoch in range(epochs):
        running_loss = 0.0
        for images in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            labels = torch.randint(0, 10, (images.size(0),))    # Fake labels
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch complete")
        sys.stdout.flush()


# Function to find optimal batch size
def optimal_batch_size(dataset, model, criterion, optimizer, starting_batch_size=64):
    batch_size = starting_batch_size
    lower_bound, upper_bound = 0, None

    while True:
        print(f"Trying to run epoch with batch size: {batch_size} @ {datetime.datetime.now()}")
        sys.stdout.flush()
        try:
            # Attempt train
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
            train(dataloader, model, criterion, optimizer, epochs=1) 

            lower_bound = batch_size  # If successful, set lower bound to current batch_size
            if upper_bound is None:  # If doubling has not failed yet, double batch size again
                batch_size *= 2
            else:  
                prev = batch_size
                batch_size = (lower_bound + upper_bound) // 2    # Binary search algorithm to find optimal batch size
                if prev == batch_size:
                    return batch_size           # If batch size doesn't change then return final batch size

        except RuntimeError as e:
            if 'memory' in str(e):
                print("Memory error")
                upper_bound = batch_size  # If fail, set upper bound to current batch size
                batch_size = (lower_bound + upper_bound) // 2
                if upper_bound - lower_bound <= 1:  
                    return lower_bound      # Return optimal batch size
            else:
                raise e     # Real error

# Transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256))])

# Download and load the training data
# Create dataset and data loader
image_dir = './data/3x256x256'  # Path to your image directory
dataset = CustomDataset(image_dir=image_dir, transform=transform)

# Initialize the model, define the loss function and the optimizer
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Find optimal batch size
starting_batch_size = 1
batch_size = optimal_batch_size(dataset, model, criterion, optimizer, starting_batch_size)
print(f"Optimal batch size: {batch_size}")

# Train
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
train(dataloader, model, criterion, optimizer, epochs=5)
print("\nFinished Training")