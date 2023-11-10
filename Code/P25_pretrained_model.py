import torch.nn
import torchvision


# pretrained = True
vgg16 = torchvision.models.vgg16(weights= torchvision.models.VGG16_Weights.DEFAULT)
# or
# vgg16 = torchvision.models.vgg16(weight='DeFault')

# pretrained = False
vgg16_false = torchvision.models.vgg16(weights=None)
# or
vgg16_false_2 = torchvision.models.vgg16()
print(vgg16)

# add module at last
# vgg16.add_module('linear', torch.nn.Linear(1000, 10))
# print(vgg16)
"""
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
  (linear): Linear(in_features=1000, out_features=10, bias=True)
)
"""

# add module at certain block
vgg16.classifier.add_module('linear in Classifier', torch.nn.Linear(1000, 10))
print(vgg16)
"""
 (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
    (linear in Classifier): Linear(in_features=1000, out_features=10, bias=True)
"""

# replace the modul
vgg16_false.classifier[6] = torch.nn.Linear(4096, 10)
print(vgg16_false)
"""
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=10, bias=True)
  )
"""