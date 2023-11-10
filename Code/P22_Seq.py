import torch
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.tensorboard import SummaryWriter
# Sequential类似封装的功能
# 不使用Sequential 的常规写法
"""
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2, stride=1)
        self.mp1 = MaxPool2d(kernel_size=2)
        self.conv2 =Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2,stride=1)
        self.mp2 = MaxPool2d(kernel_size=2)
        self.conv3 = Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2, stride=1)
        self.mp3 =MaxPool2d(kernel_size=2)
        self.linear1 = Linear(in_features=1024, out_features=64)
        self.linear2 = Linear(in_features=64, out_features=10)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.mp1(x)
        x = self.conv2(x)
        x = self.mp2(x)
        x = self.conv3(x)
        x = self.mp3(x)
        x = x.view(batch_size, -1)
        x = self.linear1(x)
        x = self.linear2(x)
        return x


model = Net()
inputs = torch.ones((64, 3, 32, 32))
outputs = model(inputs)
print(outputs.shape)  # torch.Size([64, 10])
"""


# 使用Sequential将网络结构封装起来
class Seq(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            Conv2d(3, 32, kernel_size=5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, kernel_size=5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, kernel_size=5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


model_seq = Seq()
inputs = torch.ones((64, 3, 32, 32))
outputs = model_seq(inputs)
print(outputs.shape)  # torch.Size([64, 10])

writer = SummaryWriter('../logs_seq')
writer.add_graph(model_seq, inputs)
writer.close()