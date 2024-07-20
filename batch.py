import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Define a simple feedforward neural network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 3 * 3)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(trainloader, model, criterion, optimizer, epochs):
    # Training loop
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch complete, Loss: {running_loss / len(trainloader)}")


# Function to find optimal batch size
def optimal_batch_size(trainset, model, criterion, optimizer, starting_batch_size=64):
    batch_size = starting_batch_size
    lower_bound, upper_bound = 0, None

    while True:
        print(batch_size)
        try:
            # Attempt train
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
            train(trainloader, model, criterion, optimizer, epochs=1) 

            lower_bound = batch_size  # If successful, set lower bound to current batch_size
            if upper_bound is None:  # If doubling has not failed yet, double batch size again
                batch_size *= 2
            else:  
                batch_size = (lower_bound + upper_bound) // 2    # Binary search algorithm to find optimal batch size

        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                upper_bound = batch_size  # If fail, set upper bound to current batch size
                batch_size = (lower_bound + upper_bound) // 2
                if upper_bound - lower_bound <= 1:  
                    return lower_bound      # Return optimal batch size
            else:
                raise e     # Real error

# Transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Download and load the training data
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Initialize the model, define the loss function and the optimizer
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Find optimal batch size
starting_batch_size = 1000
batch_size = optimal_batch_size(trainset, model, criterion, optimizer, starting_batch_size)
print(f"Optimal batch size: {batch_size}")

# Train
trainloader = torch.utils.data.DataLoader(trainset, batch_size=optimal_batch_size, shuffle=True)
train(trainloader, model, criterion, optimizer, epochs=10)
print("Finished Training")