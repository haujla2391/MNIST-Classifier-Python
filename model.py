import torch.nn as nn

class Model(nn.Module):
    def __init__(self):             # define layers and activation functions
        super().__init__()
        self.flatten = nn.Flatten()             # Converts (1, 28, 28) -> (784)
        self.fc1 = nn.Linear(784, 128)          # Auto includes the bias for every neuron
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):           # Connect parts and return output
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# We flatten here and not in data.py because it makes switching models easier and keeps images in original shape
# Input (batch_size, 1, 28, 28) -> Flatten (batch_size, 784) -> First Layer (batch_size, 128) -> ReLU -> 
# Second Layer (batch_size, 10)
