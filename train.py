import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from math import floor
from dataset import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.optim.lr_scheduler import StepLR
from model import CNN
from tqdm import tqdm
from pytorchtools import EarlyStopping

def train(model, device, train_loader, optimizer, epoch):
    train_losses = []
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    model.train()

    train_process = tqdm(train_loader, ascii=True)
    for _, (images, labels) in enumerate(train_process):
        images, labels = images.to(device), labels.to(device)

        # Forward + Backward + Update
        outputs = model(images)

        loss = criterion(outputs, labels)
        train_losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_process.set_description_str("Epoch: {}".format(epoch))
        train_process.set_postfix_str("loss:{:.4f}".format(loss.item()))

    return np.average(train_losses)

def test(model, device, val_loader):
    valid_losses = []

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    correct = 0
    accuracy = 0

    model.eval()

    with torch.no_grad():
        for _, (images, labels) in enumerate(val_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward + Loss
            # outputs: (B x C) B: Batch Size, C: Number of class
            outputs = model(images)

            loss = criterion(outputs, labels)
            valid_losses.append(loss.item())

            # the index for maximum probability is our predict result
            _, index = torch.max(outputs, dim=1)
            correct += torch.sum(labels == index)

    accuracy = 100. * correct.item() / len(val_loader.dataset)

    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        np.average(valid_losses), correct, len(val_loader.dataset), accuracy))

    return np.average(valid_losses)

def save_loss(path, train_loss, valid_loss):
    with open(path, 'w') as f:
        f.write('Training Loss, {}\n'.format(','.join(map(str, train_losses))))
        f.write('Validation Loss, {}\n'.format(','.join(map(str, valid_losses))))

def visualize(train_loss, valid_loss):
    plt.plot(range(1,len(train_loss)+1), train_loss, label='Training Loss', marker = 'o')
    plt.plot(range(1,len(valid_loss)+1), valid_loss, label='Validation Loss', marker = 'o')

    # find position of lowest validation loss
    min_idx = valid_loss.index(min(valid_loss)) + 1 
    plt.axvline(min_idx, linestyle='--', color='r', label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.xlim(0, len(train_loss) + 1)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    root = '../../dataset/mnist'
    batch_size = 16
    num_workers = 0
    lr = 0.001
    epoch_iter = 50
    patience = 5

    train_losses = []
    valid_losses = []

    # without data augmentation i obtained an accuracy of 0.98417
    transform_train = transforms.Compose([transforms.RandomRotation(10, fill=(0,)), 
                                          transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2), fillcolor=(0,)), 
                                          transforms.ToTensor(), 
                                          transforms.Normalize((0.1307,), (0.3081,))])

    transform_val = transforms.Compose([transforms.ToTensor(), 
                                        transforms.Normalize((0.1307,), (0.3081,))])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # define model
    model = CNN().to(device)

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # load dataset
    train_data = MNIST(root, stage='train', transform=transform_train)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, drop_last=True, num_workers=num_workers)
    
    val_data = MNIST(root, stage='validate', transform=transform_val)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=True, drop_last=True, num_workers=num_workers)

    early_stopping = EarlyStopping(patience=patience, verbose=True)
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    for epoch in range(1, epoch_iter + 1):
        avg_train_loss = train(model, device, train_loader, optimizer, epoch)
        avg_valid_loss = test(model, device, val_loader)

        train_losses.append(avg_train_loss)
        valid_losses.append(avg_valid_loss)

        # scheduler.step()

        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(avg_valid_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break

    save_loss('loss.csv', train_losses, valid_losses)

    # visualize
    visualize(train_losses, valid_losses)
