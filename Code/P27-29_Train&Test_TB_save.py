import torch
from model import Net as nn_model
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import time


transform = transforms.Compose([
    transforms.ToTensor()
    # ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.CIFAR10(root='../dataset/CIFAR-10', train=True, download=False, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = datasets.CIFAR10(root='../dataset/CIFAR-10', train=False, download=False, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = nn_model()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using Device: {device}')
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
writer = SummaryWriter('../logs_traintest_20')
train_step = 0
test_step = 0
start_time = time.time()


def train(epoch):
    running_loss = 0.0
    # 设置总minibatch数量为全局变量实现跨循环累计
    global train_step

    # 索引i代表抽取的第i个mini-batch,每个epoch更新一次
    for i, data in enumerate(train_dataloader):
        imgs, labels = data
        imgs, labels = imgs.to(device), labels.to(device)
        prediction = model(imgs)
        loss = criterion(prediction, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        train_step += 1
        train_step_counter = train_step
        # 每100个minibatch 更新图像
        if train_step_counter % 100 == 0:
            # loss over train_step
            end_time = time.time()
            print(end_time - start_time)
            writer.add_scalar('train_loss', loss.item(), train_step)

    print('epoch=', epoch+1, '  {}-th mini-batch'.format(i+1), '  loss=', running_loss/(i+1))


def test_acc():
    correct = 0
    test_loss = 0.0
    global test_step
    with torch.no_grad():
        for i, data in enumerate(test_dataloader, 0):
            imgs, targets = data
            imgs, targets = imgs.to(device), targets.to(device)
            prediction = model(imgs)
            loss = criterion(prediction, targets)

            test_loss += loss.item()
            _, predicted = torch.max(prediction, dim=1)
            correct += (predicted == targets).sum().item()
            test_step += 1
        writer.add_scalar('test_loss', test_loss, test_step)
        print('Test Accuracy= {:.3f} %'.format(100* correct/len(test_dataset)))


for epoch in range(20):
    train(epoch)
    test_acc()
    # 每个epoch保存数据 回头存下最好的数据
    # torch.save(model, '../models/...')
writer.close()
torch.save(model, "../models/traintest_{}.pth".format(epoch+1))