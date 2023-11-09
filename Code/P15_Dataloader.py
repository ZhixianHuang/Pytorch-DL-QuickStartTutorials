from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
# drop_last=True 如果最后一次抽取的样本数量不能满足一个mini-batch的batch size的要求 就会丢弃最后一次抽取的样本数量
train_data = datasets.CIFAR10(root='../dataset/CIFAR-10', train=True, download=False, transform=transform)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)

test_data = datasets.CIFAR10(root='../dataset/CIFAR-10', train=False, download=False, transform=transform)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0)

img, target = train_data[0]
print(img.shape)
print(target)

writer = SummaryWriter('../logs_CIFAR10')
step = 0
for data in test_dataloader:
    img, target = data
    writer.add_images('dataset_with_dataloader_droplast_True', img, step)
    step = step + 1
    # batch_size = 4
    #                       batch_size  channels  heigth  width
    # print(img.shape) =    [4,          3,        32,     32]
    #                    mini-batch中每个照片对应的标签对应的索引
    # print(target)    =e.g.[1,          0,         8,      8]
writer.close()
