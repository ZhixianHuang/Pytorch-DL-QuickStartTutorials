from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision import transforms
import torch

batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor()])

test_dataset = datasets.CIFAR10(root='../dataset/CIFAR-10', train=False, download=False, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)


class ConvNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # dilation 空洞卷积
        # https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0, dilation=1, )

    def forward(self, x):
        x = self.conv1(x)
        return x


step = 0
model = ConvNet()
writer = SummaryWriter('../logs_CIFAR10_Conv')
for data in test_dataloader:
    imgs, targets = data
    print(imgs.shape)  # torch.Size([64, 3, 32, 32])    [batch_size, channels, height, width]
    writer.add_images('inputs', imgs, step)
    outputs = model(imgs)
    print(outputs.shape)  # torch.Size([64, 6, 30, 30])  channels=6 无法输出对应的RGB图像, which channel=3
    # reshape( 通过扩大batch_size的方式，将一张图片由6个channel改为3个channel输出）
    # -1 占位, 表示该位置的数值自动计算  height,width 因为没有padding而减小-2
    outputs = torch.reshape(outputs, (-1, 3, 30, 30))
    writer.add_images('outputs', outputs, step)
    step = step + 1

writer.close()