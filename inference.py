import torch
import matplotlib.pyplot as plt
from dataset import MNIST
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from model import CNN

def visualize(images, pred):
    grid = make_grid(images, nrow=4)

    plt.imshow(grid.cpu().numpy().transpose((1, 2, 0)))
    plt.axis('off')
    plt.title(pred.cpu().numpy())
    plt.show()

if __name__ == "__main__":
    root = '../../dataset/mnist/digit-recognizer'
    pth_file = './checkpoint.pt'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor()])

    # load dataset
    test_data = MNIST(root, train=False, transform=transform)
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