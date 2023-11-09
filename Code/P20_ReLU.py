import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

dataset = datasets.CIFAR10(root='../dataset/CIFAR-10', train=False, download=False, transform=transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)


class Activation_ReLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # inplace 指的是是否在原址覆盖 默认为false 方便数据存储
        self.activation_ReLU = torch.nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.activation_ReLU(x)
        return x


class Activation_Sigmoid(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # inplace 指的是是否在原址覆盖 默认为false 方便数据存储
        self.activation_Sigmoid = torch.nn.Sigmoid()
        # 额外定义一个标准化(归一化),标准化可以让输入靠近激活函数活跃部分,从而避免梯度消失,每次都处在激活函数更新活跃区域
        # 输入参数取input的channel数
        # affine is a bool value
        # when set to ``True``, this module has learnable affine parameters. Default: True
        # affine=True 归一化参数 γ,β,ε为可学习参数
        self.BatchNormalization = torch.nn.BatchNorm2d(100, affine=False)

    def forward(self, x):
        x = self.activation_Sigmoid(x)
        return x




model_ReLU = Activation_ReLU()
model_Sigmoid = Activation_Sigmoid()
writer = SummaryWriter('../logs_CIFAR10_Activation')
step = 0
for data in dataloader:
    imgs, labels = data
    outputs_ReLU = model_ReLU(imgs)
    outputs_Sigmoid = model_Sigmoid(imgs)
    # ReLU produces non-negative values
    # inputs in range [0, 255] which are already non-negative
    # pictures over ReLU gives almost no differences
    writer.add_images('Activation_ReLU', outputs_ReLU, step)
    writer.add_images('Activation_Sigmoid', outputs_Sigmoid, step)
    step = step + 1


writer.close()