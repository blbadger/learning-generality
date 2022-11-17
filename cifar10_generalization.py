# cifar10_generalization.py
# MLP-style model with GPU acceleration for latent space exploration.

# import standard libraries
import time
import pathlib
import os
import pandas as pd 
import random

# import third party libraries
import numpy as np 
import torch
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader, Dataset
import torchvision
import matplotlib.pyplot as plt  
import torchvision.transforms as transforms
from google.colab import files
from google.colab import drive

# send model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print (f"Device: {device}")

drive.mount('/content/gdrive')

transform = transforms.Compose(
    [transforms.ToTensor()])

batch_size = 512
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

class FCnet(nn.Module):

    def __init__(self, starting_size):

        super().__init__()
        starting = starting_size
        self.input_transform = nn.Linear(32*32*3, starting)
        self.d1 = nn.Linear(starting, starting//2)
        self.d2 = nn.Linear(starting//2, starting//4)
        self.d3 = nn.Linear(starting//4, starting//8)
        self.d4 = nn.Linear(starting//8, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, input_tensor):
        input_tensor = torch.flatten(input_tensor, start_dim=1)
        out = self.input_transform(input_tensor)
        out = self.relu(out)

        out = self.d1(out)
        out = self.relu(out)

        out = self.d2(out)
        out = self.relu(out)

        out = self.d3(out)
        out = self.relu(out)

        out = self.d4(out)
        return out


class ConvNet(nn.Module):

    def __init__(self):

        super().__init__()
        self.conv1 = nn.Conv2d(3, 60, 5, padding=2)
        self.conv2 = nn.Conv2d(60, 80, 5, padding=2)
        self.conv3 = nn.Conv2d(80, 160, 3, padding=1)
        self.conv4 = nn.Conv2d(160, 320, 3, padding=1)
        self.dense = nn.Linear(20480, 10)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.softmax = nn.Softmax()

    def forward(self, input_tensor):
        out = self.conv1(input_tensor)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.conv4(out)
        out = self.relu(out)

        out = torch.flatten(out, start_dim=1)
        out = self.dense(out)
        out = self.softmax(out)
        return out


def loss_gradient(model, input_tensor, target_output, output_dim):
    """
     Computes the gradient of the input wrt. the objective function

     Args:
        input: torch.Tensor() object of input
        model: Transformer() class object, trained neural network

     Returns:
        gradientxinput: arr[float] of input attributions

    """

    # change output to float
    target_output = target_output.reshape(1)
    input_tensor.requires_grad = True
    output = model.forward(input_tensor)

    loss = loss_fn(output, target_output)

    # backpropegate output gradient to input
    loss.backward(retain_graph=True)
    gradient = input_tensor.grad

    return gradient


def show_batch(input_batch, count=0, grayscale=False):
    """
    Show a batch of images with gradientxinputs superimposed

    Args:
        input_batch: arr[torch.Tensor] of input images
        output_batch: arr[torch.Tensor] of classification labels
        gradxinput_batch: arr[torch.Tensor] of attributions per input image
    kwargs:
        individuals: Bool, if True then plots 1x3 image figs for each batch element
        count: int

    returns:
        None (saves .png img)

    """

    plt.figure(figsize=(15, 15))
    for n in range(16*16):
        ax = plt.subplot(16, 16, n+1)
        plt.axis('off')
        if grayscale:
            plt.imshow(input_batch[n], cmap='gray_r')
        else:
            plt.imshow(input_batch[n])
        plt.tight_layout()

    plt.tight_layout()
    plt.show()
    plt.savefig('gan_set{0:04d}.png'.format(count), dpi=410)
    plt.close()
    return

def train_model(model, optimizer, loss_fn, epochs, size='2k'):
    """
    Train the model using gradient splitting.

    Args:
        model: torch.nn object
        optimizer: torch.nn optimizer
        loss_fn: torch.optim object
        epochs: int, number of desired training epochs
    kwargs:
        size: str, one of '2k', '5k', '10k', '20k', size of training
            ` and test data

    Returns:
        None (modifies model in-place, prints training curve data)

    """
    model.train()
    count = 0
    total_loss = 0
    start = time.time()
    train_array, test_array = [], []

    for e in range(epochs):
        total_loss = 0
        count = 0

        for pair in zip(trainloader, testloader):
            train_x, train_y, test_x, test_y = pair[0][0], pair[0][1], pair[1][0], pair[1][1]
            count += 1

            # assumes minibatch size of 512
            if (size == '2k' and count > 10) or (size == '5k' and count > 20) or (size == '10k' and count > 40) or (size == '20k' and count > 80):
                break

            trainx = train_x.to(device)
            output = model(trainx)
            loss = loss_fn(output.to(device), train_y.to(device))
            loss = loss.to(device)
            test_x = test_x.to(device)
            output2 = model(test_x)
            loss -= loss_fn(output2.to(device), test_y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss

        ave_loss = float(total_loss) / count
        elapsed_time = time.time() - start
        print (f"Average Loss: {ave_loss:.04}")
        train_array.append(test_model(trainloader, model))
        test_array.append(test_model(testloader, model))
        start = time.time()

    print (train_array, test_array)

    return

def test_antitrain(test_dataloader, model):
    model.train()
    count = 0
    total_loss = 0
    start = time.time()

    for batch, (x, y) in enumerate(test_dataloader):
        if count > 10:
            break
        count += 1
        x = x.to(device)
        output = model(x)
        loss = -loss_fn(output.to(device), y.to(device))
        loss = loss.to(device)
        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()
        total_loss += loss

    ave_loss = float(total_loss) / count
    elapsed_time = time.time() - start
    start = time.time()
    return

def test_model(test_dataloader, model):
    model.eval()
    correct, count = 0, 0
    batches = 0
    for batch, (x, y) in enumerate(test_dataloader):
        if batch > 20:
            break
        x = x.to(device)
        predictions = model(x)
        _, predicted = torch.max(predictions.data, 1)
        count += len(y)
        correct += (predicted == y.to(device)).sum().item()
        batches += 1

    print (f'Accuracy: {correct / count}')
    return correct / count


train_accuracies, test_accuracies = [], []
torch.cuda.empty_cache()
epochs = 200
loss_fn = nn.CrossEntropyLoss()
model = ConvNet()

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print (total_params)

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
train_model(model, optimizer, loss_fn, epochs)
trainloader = trainloader
testloader = testloader
print ('end')
train_accuracies.append(test_model(trainloader, model))
test_accuracies.append(test_model(testloader, model))

print (train_accuracies)
print (test_accuracies)

data_dir = '/content/gdrive/path_to_dir'
torch.save(model.state_dict(), data_dir)
