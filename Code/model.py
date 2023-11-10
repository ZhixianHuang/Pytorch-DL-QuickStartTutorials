import torch
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    nn = Net()
    inputs = torch.ones((64, 3, 32, 32))
    outputs = nn(inputs)
    print(outputs.shape)
    # torch.Size([64, 10])
    # test_passed