from torch.utils.tensorboard import SummaryWriter
import cv2
from PIL import Image
import numpy as np

writer = SummaryWriter("logs")
for i in range(100):
    writer.add_scalar("y = 2x", 2*i, i)


# writer.add_image(tag, img_tensor, global_step, ... , dataformats="CHW")
# img_tensor需要接受张量torch.Tensor 或者np.array
# 转换为np.array有两种方式 使用opencv 或者 numpy


# 1 使用openCV
img_path = r'dataset/exercise_data/train/ants_image/0013035.jpg'
img_array_cv = cv2.imread(img_path)
print(type(img_array_cv))
writer.add_image('CV', img_array_cv, 3, dataformats="HWC")

# 2 使用numpy
img_path = r'dataset/exercise_data/train/ants_image/0013035.jpg'
img_PIL = Image.open(img_path)
img_array_np = np.array(img_PIL)
print(type(img_array_np))

# global_step 理解见Transforms.py
# 同时 函数默认的输入格式是 3*H*W 所以需要调整输入
# 使用img_PIL.shape 读取原始形状 发现是按照(H,W,3) 也就是(H,W,C)
print(img_array_np.shape)
writer.add_image('test', img_array_np, 5, dataformats="HWC")
writer.close()

# CV2 和 Numpy保存的通道RGB顺序不同
# CV2是BGR顺序
# Numpy是RGB顺序