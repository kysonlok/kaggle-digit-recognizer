import torch
import matplotlib.pyplot as plt
from dataset import MNIST
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from model import CNN

if __name__ == "__main__":
    root = '../../dataset/mnist/digit-recognizer'
    pth_file = './checkpoint.pt'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor()])

    # 加载数据集
    test_data = MNIST(root, train=False, transform=transform)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
    
    # 定义模型
    model = CNN().to(device)
    model.eval()
    model.load_state_dict(torch.load(pth_file))

    for _, images in enumerate(test_loader):
        images = images.to(device)

        outputs = model(images)
        _, index = torch.max(outputs, dim=1)

        grid = make_grid(images, nrow=4)

        plt.imshow(grid.cpu().numpy().transpose((1, 2, 0)))
        plt.axis('off')
        plt.title(index.cpu().numpy())
        plt.show()
        break