from torch.utils.data import Dataset
import os
from PIL import Image


# 一个图片的硬地址构成:
# E:\PyTorch深度学习快速入门教程-我是土堆\dataset\hymenoptera_data\train\ants\         0013035.jpg
# -------------------------root_dir--------------------------------|label_dir|----image_name---|
#                          数据集位置                                   标 签          图片名
# 也就是图片的上一级就是标签，这就是为什么把(path) root_dir和label_dir通过os拼接起来的原因

class MyData(Dataset):
    def __init__(self, root_dir: object, label_dir: object) -> object:
        #  E:\PyTorch深度学习快速入门教程-我是土堆\dataset\hymenoptera_data\train 数据集位置
        self.root_dir = root_dir
        # 数据集中包含的标签目录中的具体一个标签
        self.label_dir = label_dir
        # 训练集该标签下所有的文件名
        # E:\PyTorch深度学习快速入门教程-我是土堆\dataset\hymenoptera_data\train\ants 标签下所有存储的图片
        # 拼接数据集和标签目录成完整路径
        self.path = os.path.join(self.root_dir, self.label_dir)
        # 打包放进一个列表中 目的是之后可以通过使用索引获取对应索引下图片的文件名
        self.image_path = os.listdir(self.path)

    def __getitem__(self, index):
        # 目标图片的下标
        image_name = self.image_path[index]
        # 构建图图片的硬地址完整路径 数据集+标签+图片文件名
        image_item_path = os.path.join(self.path, image_name)
        # 加载图像
        img = Image.open(image_item_path)
        label = self.label_dir
        # 返回图片与其对应标签
        return img, label

    def __len__(self):
        # 该标签下有多少照片
        return len(self.image_path)


root_dir = r'dataset/hymenoptera_data/train'
ants_label_dir = 'ants'
bees_label_dir = 'bees'
ants_dataset = MyData(root_dir, ants_label_dir)
bees_dataset = MyData(root_dir, bees_label_dir)

ants_img, ants_label = ants_dataset[1]
bees_img, bees_label = bees_dataset[1]
print(ants_label)
print(ants_dataset.__len__())  # 124
print(bees_dataset.__len__())  # 121
ants_img.show()
bees_img.show()

test_dataset = ants_dataset + bees_dataset
print(len(test_dataset))
test_img, test_label = test_dataset[1]
test_img.show()

'''
image_path = r'E:\PyTorch深度学习快速入门教程-我是土堆\dataset\hymenoptera_data\train\ants\0013035.jpg'
image = Image.open(image_path)
print(image.size)
image.show()
'''