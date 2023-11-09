from torchvision import datasets
from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


# 无Transform 直接展示图片
"""
CIFAR_train_dataset = datasets.CIFAR10(root='../dataset/CIFAR-10', train=True, download=False)
CIFAR_test_dataset = datasets.CIFAR10(root='../dataset/CIFAR-10', train=False, download=False)

print(CIFAR_test_dataset[0])
print(CIFAR_test_dataset.classes)
image , target = CIFAR_test_dataset[0]
print(CIFAR_test_dataset.classes[target])
image.show()
"""

# 有transform 将图片压成Tensor后用tensorboard输出
transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

train_set = datasets.CIFAR10(root='../dataset/CIFAR-10', train=True, download=True, transform=transform)
test_set = datasets.CIFAR10(root='../dataset/CIFAR-10', train=False, download=True, transform=transform)

print(test_set[0])
writer = SummaryWriter('../logs_CIFAR')
for i in range(15):
    image, target = test_set[i]
    writer.add_image('CIFAR', image, i)
writer.close()