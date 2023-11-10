import torch
from torchvision import transforms
from torch.nn import Conv2d, MaxPool2d, Linear, Flatten
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

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


dataset = datasets.CIFAR10(root='../dataset/CIFAR-10', train=False, download=False, transform=transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

model_seq = Seq()
model_seq.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_seq.parameters(), lr=0.01)


epoch_list = []
loss_list = []
for epoch in range(30):
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        img, label = data
        img, label = img.to(device), label.to(device)

        outputs = model_seq(img)
        loss = criterion(outputs, label)
        running_loss = running_loss + loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('epoch=', epoch, 'loss=', running_loss/64)
    epoch_list.append(epoch)
    loss_list.append(running_loss/64)

plt.plot(epoch_list, loss_list)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()