import torch
import torchvision
# solution 2
from P26_model_save import *

model = torch.load('../models/vgg16_with_pretrained.pth')
# model is saved with an additional linear layer on the end
print(model)  # (Linear): Linear(in_features=1000, out_features=10, bias=True) exists

# model now has two linear layers at the end
# model_statedict = torch.load('../models/vgg16_with_pretrained_statedict.pth')
# 发现获取的全都是字典形式 打印模型不再是网络格式
# print(model_statedict)

# 2.Option的获取方式
# 首先创建一个没有预训练的神经网络结构(空壳）
vgg16_network_without_pretrained = torchvision.models.vgg16()
# 获取预训练数据的词典
models_statedict = torch.load('../models/vgg16_with_pretrained_statedict.pth')
# 将包含训练数据的词典放进空网络结构中去，就变成了一个包含训练数据的网络
vgg16_network_without_pretrained_plus_dict = vgg16_network_without_pretrained.load_state_dict(models_statedict)
print(vgg16_network_without_pretrained_plus_dict)


# solution 1
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x


model_single_conv = torch.load('../models/single_conv.pth')
print(model_single_conv)
# Error: AttributeError: Can't get attribute 'Net' on <module '__main__' from 'E:\\PyTorch-DL-QuickStartTutorial\\logs_CIFAR\\P26_model_load.py'>

# successfully loaded
#Net(
#   (conv): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# )