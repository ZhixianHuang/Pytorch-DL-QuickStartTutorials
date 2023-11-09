
# Transforms是什么？
# transforms是一个用于数据预处理的模块，它提供了许多常用的图像预处理和数据增强的操作。
# transforms可以用来对图像、文本、音频等数据进行各种操作，以便更好地适应于深度学习模型的训练
# 在图像处理中，transforms常用于对训练数据进行一系列的预处理，如调整图像大小、随机裁剪、翻转、标准化等，以增强模型的训练效果。

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

img_path = r'dataset/torch.jpg'
img = Image.open(img_path)

writer = SummaryWriter('logs')

# ToTensor
# 为什么需要Tensor
# Tensor有一些属性，比如反向传播、梯度等属性，它包装了神经网络需要的一些属性
# 实例化transformers.ToTensor
# transformers.ToTensor 自带__call__函数
# __call__是Python中的一个特殊方法（也叫魔法方法），当在一个对象上调用()运算符时，会自动执行__call__方法。这使得可以像调用函数一样来调用一个类的实例
transform = transforms.ToTensor()
img_tensor = transform(img)
# print(img_tensor.shape)
writer.add_image('pytorch_before_norm', img_tensor)


# Normalize
# output[channel] =  (input[channel] - mean[channel]) /   std[channel]
# RGB image, channel = 3
transform_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_tensor = transform_norm(img_tensor)
print(img_tensor.shape)
writer.add_image('pytorch_after_norm', img_tensor)

# Resize
print(img)  # 打印img信息 能够获取img的数据类型与尺寸 PIL image 3*1100*619
transform_resize = transforms.Resize((512, 512))
# Resize直接接受image
img_resize = transform_resize(img)
# 裁剪尺寸后Resize
img_resize_tensor = transform(img_resize)
print(img_resize_tensor.shape)
writer.add_image('image_resize', img_resize_tensor)

# Compose (resize)
print(img) # img 在导入之前的shape是WHC mode=RGB size=1100x619
# 导入pytorch的shape是 CHW
# Resize(size)只接收一个int的话 会按照 [将短边缩小为size,并等比将长边缩放]
# if height>width (width为短边=size) output.shape = (size*height/width, size)
# 短边缩放ratio = width/size  长边height缩放为 height/(width/size) = height*size/width
# 同理 if height<width (height=size) output.shape=(size, size*width/height)
transform_resize_2 = transforms.Resize(128)
# 函数级联, 打包一系列transform实例的操作
# 使用compose时候一定要关注每一个操作的输入与输出, 前序输出要和后续输入的维度匹配
# PIL  --Resize--  PIL  --ToTensor--  Tensor
transform_compose = transforms.Compose([
                    transform_resize_2, transform  # Resize + ToTensor
                    ])
img_compose_resize = transform_compose(img)
writer.add_image('image_resize_2', img_compose_resize)
print(img_compose_resize.shape)
# height.old=619 height_new=128 ratio=619/128=4.836
# width_old=1100 width_new=1100/4.836=227


# RandomCrop
# global_step可以理解为存储的张书
# 在tag不变的情况下 global_step=4 可以同时记录下四张照片
transform_RandCrop = transforms.RandomCrop(256)
transform_compose_randcrop = transforms.Compose([
    transform_RandCrop,
    transform
])
img_randcrop = transform_compose_randcrop(img)
writer.add_image('image_RandomCrop', img_randcrop, global_step=4)

# 或是随机截取10涨
for i in range(10):
    img_randcrop_loop = transform_compose_randcrop(img)
    writer.add_image('image_RandomCrop_Loop', img_randcrop_loop, i)
writer.close()
