import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Batch: a small subset of dataset used for training in one forward and backward pass
# Iteration: 1 forward and backward pass
# Epoch: 1 complete pass of entire traning dataset

# Small batch size (32, 64) generalizes better on unseen data and can be more fitting to noise, large batch size (512, 1024)
# may require lots of memory but updates are more smooth and stable. Situations for both.

# More epochs means better E_in but worse E_out and overfitting if too large

def get_dataloaders(batch_size: int):
    """
    Returns:
        train_loader
        test_loader
    """
    # compose takes a list of Transform objects and composes them together
    transform = transforms.Compose([
        # Converts PIL Image (0–255) -> Float Tensor (0–1) and normalizes by doing (x - mean) / (stdev)
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 60k for training, 10k for testing
    train_data = datasets.MNIST(root="./data", train=True, transform=transform, download=True)  
    test_data = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    # Data Loaders which group the data into batches, shuffles in training so we don't learn the sequence instead of features
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False) # false because testing is like real time sequential data 

    return train_loader, test_loader

# if __name__ == "__main__":
#     batchSize = 64
#     train_loader, test_loader = get_dataloaders(batchSize)

#     images, labels = next(iter(train_loader))
#     print(images.shape)     # [64, 1, 28, 28]   this is 64 images in the batch
#     print(labels.shape)     # [64]


# We normalize because Neural Networks learn much faster when input is zero centered, it is centered around the mean.
# Also, it prevents our tanh or sigmoid functions from becoming near 0, due to larger values, which hinders learning.
# It presents all dimensions on equal footing. The pixels are all represented evenly (brighter and duller same footing).

# The images are shaped (1,28,28) because the 1 is a color channel which for MNIST is just one grayscale value not RGB, etc

# Flattening an image is where we convert to 1D array for our layers, we don't do it here because we aren't iterating 
# through them and we haven't built our model yet
