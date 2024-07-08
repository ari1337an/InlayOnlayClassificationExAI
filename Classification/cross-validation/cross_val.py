import os
import gc
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from pytorch_pretrained_vit import ViT
import re
from collections import defaultdict

torch.manual_seed(20)

device = "cpu"
print(f"Using {device} device")

# Directory Names
dir_dataset = './dataset/dataset_stl-to-voxel_tooth_npys_70x70x70/'

class ToothDataset(Dataset):
    def __init__(self, img_dir, files, transform=None):
        self.dataset_path = img_dir
        self.transform = transform
        self.npy_files = files

    def __len__(self):
        return len(self.npy_files)

    def __getitem__(self, idx):
        if idx >= len(self.npy_files):
            print("No datafile/image at index: " + str(idx))
            return None
        
        npy_filename = self.npy_files[idx]
        label = int('onlay' in npy_filename)
        numpy_arr = np.load(os.path.join(self.dataset_path, npy_filename), allow_pickle=True)

        # Ensure numpy array has correct dimensions
        if numpy_arr.shape[0] > 70:
            numpy_arr = numpy_arr[-70:]
        
        numpy_arr = numpy_arr.reshape(1, 70, 70, 70)
        tensor_arr = torch.from_numpy(numpy_arr).to(torch.float32)

        del numpy_arr
        gc.collect()

        if self.transform:
            tensor_arr = self.transform(tensor_arr)  # Apply transformations

        return tensor_arr.to(torch.float32), torch.LongTensor([label])

# Regular expression to match filenames like T1.1.npy and T4.1 (onlay).npy
pattern = re.compile(r'^T(\d+)\.\d+.*\.npy$')

# Read all file names from the dataset directory
files = [f for f in os.listdir(dir_dataset) if f.endswith('.npy')]

# Organize files by original instance
file_dict = defaultdict(list)
for file in files:
    match = pattern.match(file)
    if match:
        original_instance = int(match.group(1))  # Capture the original instance number
        file_dict[original_instance].append(file)

# Sort the lists for consistency
for key in file_dict:
    file_dict[key].sort()

# Define the folds
folds = {
    1: [1, 4],
    2: [2, 5],
    3: [3, 6]
}

# Download the pretrained weights
pretrained_vit = ViT('B_16_imagenet1k', pretrained=True)

# Freeze the ViT
for param in pretrained_vit.parameters():
    param.requires_grad = False

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # The CNN layers with maxpool and ReLU
        self.features = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),
            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),
            nn.Conv3d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),
            nn.Flatten(),
        )

        # Pretrained ViT
        self.vit = pretrained_vit

        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1000,512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512,2)
        )

    def forward(self, x):
        x = self.features(x)

        # Reshape the CNN output (output of size batch*4096) into batch*3x384x384
        x = x.repeat(1, 36, 1, 1, 1)
        x = x.view(-1, 1, 384, 384)
        x = x.repeat(1, 3, 1, 1)

        # Pass the reshaped CNN output into ViT
        x = self.vit(x)

        # Pass it into the fully connected layers
        x = self.fc(x)
        return x

# Hyperparameters
epochs = 15
batch_size = 2
learning_rate = 1e-6
weight_decay = 0.1

loss_function = nn.CrossEntropyLoss()

# The training function
def train(dataloader, model, loss_fn, optimizer):
    torch.cuda.empty_cache()
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y.squeeze())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 5 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        del pred
        del loss

# Keep record of the validation accuracy
validation_accuracy = []

# The validation function
def validation(dataloader, model, loss_fn):
    global validation_accuracy
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y.squeeze()).item()
            correct += (torch.argmax(pred, dim=1) == y.squeeze()).sum().item()
            X.cpu()
            y.cpu()
    test_loss /= num_batches
    correct /= size

    # Keep record of the validation accuracy globally
    validation_accuracy.append(correct * 100)
    # Print
    print(f"Validation Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print(f"Current best validation accuracy: {max(validation_accuracy)}%")

# Cross-validation
for fold, val_instances in folds.items():
    val_files = []
    train_files = []

    # Get validation files
    for val_instance in val_instances:
        val_files.extend(file_dict[val_instance])
    
    # Get training files
    for train_instance in file_dict.keys():
        if train_instance not in val_instances:
            train_files.extend(file_dict[train_instance])

    # Create datasets for training and validation
    training_data = ToothDataset(img_dir=dir_dataset, files=train_files, transform=None)
    validation_data = ToothDataset(img_dir=dir_dataset, files=val_files, transform=None)

    training_data_loader = DataLoader(training_data, batch_size, shuffle=True)
    validation_data_loader = DataLoader(validation_data, batch_size, shuffle=False)

    model = NeuralNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training and validation for the current fold
    print(f"Fold {fold}\n-------------------------------")
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(training_data_loader, model, loss_function, optimizer)
        validation(validation_data_loader, model, loss_function)
    print("Done with fold", fold)
