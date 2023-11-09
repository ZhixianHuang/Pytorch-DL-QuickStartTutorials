import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

test_dataset = datasets.CIFAR10(root='../dataset/CIFAR-10', train=False, download=False, transform=transforms.ToTensor())
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class MAXPOOL(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # default ceil_mode=False
        # ceil_mode=True , use ceil 向上取整
        # ceil_mode=False, use floor 向下取整
        self.mp = torch.nn.MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, x):
        x = self.mp(x)
        return x


model = MAXPOOL()
writer = SummaryWriter('../logs_CIFAR10_MP')
step = 0
for data in test_dataloader:
    imgs, targets = data
    outputs = model(imgs)
    writer.add_images('inputs', imgs, step)
    writer.add_images('outputs', outputs, step)
    step = step + 1
writer.close()