import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision import transforms

dataset = datasets.CIFAR10(root='../dataset/CIFAR-10', train=False, download=False, transform=transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)


class Linear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #  batch_size  channel   height    width
        # [64,         3,        32,       32]
        # 64*3*32*32 = 196608
        self.linear = torch.nn.Linear(196608, 10)

    def forward(self, x):
        x = self.linear(x)
        return x


model = Linear()
for data in dataloader:
    imgs, tragets = data
    # print(imgs.shape)  # torch.Size([64, 3, 32, 32])
    # 或者可以用reshape()
    # 全线性连接 batch_size=1, channel=1, height=1 , width=
    # why is the width?
    # outputs_1 = torch.reshape(imgs, (1, 1, 1, -1))
    outputs_2 = torch.flatten(imgs)
    # print(outputs_1 == outputs_2) True
    # print(outputs.shape)  # torch.Size([196608])
    # outputs = model(outputs)
    # print(outputs.shape)  # torch.Size([10])
