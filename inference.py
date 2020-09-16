import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset import MNIST
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from model import CNN
from PIL import Image

def visualize(images, pred):
    grid = make_grid(images, nrow=4)

    plt.imshow(grid.cpu().numpy().transpose((1, 2, 0)))
    plt.axis('off')
    plt.title(pred.cpu().numpy())
    plt.show()

def real_predict(model, transform, file):
    threshold = 50
    
    try:
        img = Image.open(file).convert('L')
    except IOError:
        print('Unable to load image')
        sys.exit(1)

    img = img.resize((28, 28))
    img = np.array(img)

    for i in range(28):
        for j in range(28):
            img[i][j] = 255 - img[i][j]
            if (img[i][j] < threshold):
                img[i][j] = 0
            else:
                img[i][j] = 255

    img = transform(img)
    img = img.unsqueeze(0)

    output = model(img)
    _, index = torch.max(output, dim=1)

    print('Predict result: {}'.format(index.item()))

if __name__ == "__main__":
    root = '../../dataset/mnist'
    pth_file = './checkpoint.pt'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize((0.1307,), (0.3081,))])

    # load dataset
    test_data = MNIST(root, stage='test', transform=transform)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    
    # define model
    model = CNN().to(device)
    model.eval()
    model.load_state_dict(torch.load(pth_file))

    with open('submission.csv', 'w') as f:
        f.write('ImageId,Label\n')

    for i, images in enumerate(test_loader):
        images = images.to(device)

        outputs = model(images)
        _, index = torch.max(outputs, dim=1)

        with open('submission.csv', 'a') as f:
            f.write('{},{}\n'.format(i+1, index.item()))
        
        # visualize(images, index)

    # real_predict(model, transform, 'test.jpg')