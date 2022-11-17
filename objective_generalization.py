# objective_generalization.py

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


# send model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print (f"Device: {device}")

class ImageDataset(Dataset):
    """
    Creates a dataset from images classified by folder name.  Random
    sampling of images to prevent overfitting
    """

    def __init__(self, img_dir, transform=None, target_transform=None, image_type='.png'):
        # specify image labels by folder name 
        self.img_labels = [item.name for item in data_dir.glob('*')]

        # construct image name list: randomly sample 400 images for each epoch
        images = list(img_dir.glob('*/*' + image_type))
        random.shuffle(images)
        self.image_name_ls = images[:800]

        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_name_ls)

    def __getitem__(self, index):
        # path to image
        img_path = os.path.join(self.image_name_ls[index])
        image = torchvision.io.read_image(img_path) # convert image to tensor of ints , torchvision.io.ImageReadMode.GRAY
        image = image / 255. # convert ints to floats in range [0, 1]
        image = torchvision.transforms.Resize(size=[28, 28])(image) 

        # assign label to be a tensor based on the parent folder name
        label = os.path.basename(os.path.dirname(self.image_name_ls[index]))

        # convert image label to tensor
        label_tens = torch.tensor(self.img_labels.index(label))
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label_tens


transform = transforms.Compose(
    [transforms.ToTensor()])

batch_size = 512
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
class ConvForward(nn.Module):

    def __init__(self, starting_size):

        super().__init__()
        starting = starting_size
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

class ConvBackward(nn.Module):

    def __init__(self, starting_size):

        super().__init__()
        starting = starting_size
        self.conv1 = nn.Conv2d(3, 20, 5, padding=2)
        self.conv2 = nn.Conv2d(20, 40, 5, padding=2)
        self.conv3 = nn.Conv2d(40, 80, 3, padding=1)
        self.conv4 = nn.Conv2d(80, 160, 3, padding=1)
        self.dense = nn.Linear(163840, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, input_tensor):
        out = self.conv1(input_tensor)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.relu(out)

        out = self.conv4(out)
        out = self.relu(out)

        out = torch.flatten(out, start_dim=1)
        out = self.dense(out)
        out = self.softmax(out)
        return out

def loss_gradient(model, input_tensor, true_output, output_dim):
    """
     Computes the gradient of the input wrt. the objective function

     Args:
        input: torch.Tensor() object of input
        model: Transformer() class object, trained neural network

     Returns:
        gradientxinput: arr[float] of input attributions

    """

    # change output to float
    true_output = true_output.reshape(1)
    input_tensor.requires_grad = True
    output = model.forward(input_tensor)

    loss = loss_fn(output, true_output)

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

def train_model(dataloader, model, optmizer, loss_fn, epochs):
    model.train()
    count = 0
    total_loss = 0
    start = time.time()
    train_array, test_array, loss_array = [], [], []
    test_arr2 = []
    norms = []
    for e in range(epochs):
        total_loss = 0
        count = 0
        # print (e)

        for pair in zip(trainloader, testloader):
            if count == 0:
                test_accuracy = test_model(testloader, model)
                low = test_accuracy < 0.11
                high = test_accuracy > 0.11
            train_x, train_y, test_x, test_y = pair[0][0], pair[0][1], pair[1][0], pair[1][1]
            count += 1
            if count > 20:
                break
            trainx = train_x.to(device)
            output = model(trainx)
            loss = loss_fn(output.to(device), train_y.to(device))
            test_x = test_x.to(device)
            output2 = model(test_x)
            if not low:
                loss -= loss_fn(output2.to(device), test_y.to(device))
            loss = loss.to(device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss

        norms.append(float(torch.norm(model.conv1.weight.grad)))
        print (torch.norm(model.conv1.weight.grad))
        ave_loss = float(total_loss) / count
        elapsed_time = time.time() - start
        print (f"Average Loss: {ave_loss:.04}")
        train_array.append(test_model(trainloader, model))
        test_array.append(test_model(testloader, model))
        loss_array.append(ave_loss)
        start = time.time()
        
    print (f'Train array: {train_array}')
    print (f'Test array: {test_array}') 
    print (f'Loss array: {loss_array}')
    print (f'Gradient Norms: {norms}')
    return

def test_model(test_dataloader, model):
    model.eval()
    correct, count = 0, 0
    batches = 0
    for batch, (x, y) in enumerate(test_dataloader):
        if batches > 20:
            break
        x = x.to(device)
        predictions = model(x)
        _, predicted = torch.max(predictions.data, 1)
        count += len(y)
        correct += (predicted == y.to(device)).sum().item()
        batches += 1

    print (f'Accuracy: {correct / count}')
    return correct / count

class NewResNet(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.softmax = nn.Softmax()
        self.conv1 = self.model.conv1


    def forward(self, x):
        x = self.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        x = self.softmax(x)
        return x


new_trainset = []
new_testset = []
fraction = 0 # fraction of input non-random
for i in range(len(trainset)):
    new_trainset.append(tuple([trainset[i][0] * fraction + (torch.randn(3, 32, 32)/3 + 0.5) * (1 - fraction), trainset[i][1]]))

for i in range(len(testset)):
    new_testset.append(tuple([testset[i][0] * fraction + (torch.randn(3, 32, 32)/3 + 0.5) * (1 - fraction), testset[i][1]]))

for i in range(len(trainset)//2):
    new_trainset.append(tuple([trainset[i][0] * fraction + (torch.randn(3, 32, 32)/3 + 0.5) * (1 - fraction), trainset[i][1]]))

for i in range(len(trainset)//2, len(trainset)):
    new_testset.append(tuple([trainset[i][0] * fraction + (torch.randn(3, 32, 32)/3 + 0.5) * (1 - fraction), trainset[i][1]]))

trainloader = torch.utils.data.DataLoader(new_trainset, batch_size=batch_size, shuffle=False)
testloader = torch.utils.data.DataLoader(new_testset, batch_size = batch_size, shuffle=False)

torch.cuda.empty_cache()
epochs = 2000
loss_fn = nn.CrossEntropyLoss()
model = ConvForward(8000)
# resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
# model = NewResNet(resnet)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print (total_params)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.000005)
train_model(trainloader, model, optimizer, loss_fn, epochs)
trainloader = trainloader
testloader = testloader


data_dir = '/content/gdrive/My Drive/path/to/model'
torch.save(fcnet.state_dict(), data_dir)
