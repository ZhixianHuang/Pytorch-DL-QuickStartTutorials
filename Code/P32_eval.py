import torch

from model import *
from PIL import Image
from torchvision import transforms

image_path = '../dataset/horse.jpg'
image = Image.open(image_path)
image = image.convert('RGB')
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'{device}')
image = transform(image)
network = torch.load('../models/traintest_20.pth')
image = torch.reshape(image, (1, 3, 32, 32))
image = image.to(device)
# RuntimeError: Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same or input should be a MKLDNN tensor and weight is a dense tensor
# which means the inputdata(image) is in CPU and the model(network) is on GPU
network.eval()
with torch.no_grad():
    output = network(image)
print(output)
print(output.argmax(axis=1))
# tensor([7], device='cuda:0')