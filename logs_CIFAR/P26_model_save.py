import torch
import torchvision

vgg16 = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
vgg16.add_module('Linear', torch.nn.Linear(1000, 10))
# model is saved with an additional linear layer on the end
torch.save(vgg16, '../models/vgg16_with_pretrained.pth')

# no additional layers at the end
vgg16 = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
# recommended saving option has a less memory
torch.save(vgg16.state_dict(), '../models/vgg16_with_pretrained_statedict.pth')
print(vgg16)

# trap
# 当定义了一个新的网络类并实例化后，保存该网络
# 在别的代码中想要读取这个网络，但是由于在新代码中没有定义这个类，所以直接实例化了网络是不可行的
# 两种解决方法
# 第一种：在新的代码中 也创建这个类
# 第二种：将创建该类的代码作为库导入

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x

single_conv = Net()
torch.save(single_conv, '../models/single_conv.pth')