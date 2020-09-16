import os
import pandas as pd
import numpy as np
from math import floor
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch
import torchvision

class MNIST(Dataset):
    def __init__(self, root, stage='train', transform=None):
        self.data = None
        self.stage = stage

        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])

        if stage == 'train':
            self.data = pd.read_csv(os.path.join(root, 'train.csv'))
            idx = floor(0.8*len(self.data))
            self.data = self.data.iloc[:idx, :]
        elif stage == 'validate':
            self.data = pd.read_csv(os.path.join(root, 'train.csv'))
            idx = floor(0.8*len(self.data))
            self.data = self.data.iloc[idx:, :]
        else:
            self.data = pd.read_csv(os.path.join(root, 'test.csv'))

    def get_train(self, index):
        # dtype mush be uint8, otherwise transforms.ToTensor() won't normalize to [0, 1]
        image = np.array(self.data.iloc[index, 1:].values.astype(np.uint8).reshape((28, 28)))
        label = self.data.iloc[index, 0]

        image = Image.fromarray(image, mode='L')

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def get_test(self, index):
        # dtype mush be uint8, otherwise transforms.ToTensor() won't normalize to [0, 1]
        image = np.array(self.data.iloc[index, :].values.astype(np.uint8).reshape((28, 28)))

        if self.transform is not None:
            image = self.transform(image)

        return image
       
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        '''
        # dtype mush be uint8, otherwise transforms.ToTensor() won't normalize to [0, 1]
        image = np.array(self.data.iloc[index, 1:].values.astype(np.uint8).reshape((28, 28)))
        label = self.data.iloc[index, 0]

        if self.transform is not None:
            image = self.transform(image)
        '''
        if self.stage == 'train' or self.stage == 'validate':
            return self.get_train(index)
        else:
            return self.get_test(index)

if __name__ == "__main__":
    # ToTensor()   
    '''Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8
    '''

    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    transform = transforms.Compose([transforms.ToTensor()])

    train_data = MNIST('../../dataset/mnist', stage='train', transform=transform)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

    val_data = MNIST('../../dataset/mnist', stage='validate', transform=transform)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=True)

    print(len(train_loader), len(val_loader))

    # Plotting 4x4 grid of images using matplotlib
    train_iter = iter(train_loader)
    images, labels = train_iter.next()

    grid = make_grid(images, nrow=4)

    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.axis('off')
    plt.title(labels.numpy())
    plt.show()