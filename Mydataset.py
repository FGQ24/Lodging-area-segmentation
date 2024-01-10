# # # 别动啦！！！
import os
import math
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


def convert_nonzero_to_one(array):
    # 创建一个与输入数组相同形状的新数组
    result_array = np.zeros_like(array)
    # 将非零元素变为1
    result_array[array != 0] = 1
    return result_array


class MyDataset(Dataset):
    def __init__(self, root_dir, img_dir, label_dir, transformimg, transformlab):
        # 根文件路径
        self.root_dir = root_dir
        # 图片文件路径
        self.img_dir = img_dir
        # 标签文件夹路径
        self.label_dir = label_dir
        # transform
        self.transformimg = transformimg
        self.transformlab = transformlab
        # 获取图片文件夹路径并生成图片名称的列表
        self.img_path = os.path.join(self.root_dir, self.img_dir)
        self.img_list = os.listdir(self.img_path)
        # 获取标签文件夹路径并生成标签名称的列表
        self.label_path = os.path.join(self.root_dir, self.label_dir)
        self.label_list = os.listdir(self.label_path)

    def __getitem__(self, item):
        img_name = self.img_list[item]
        img_item_path = os.path.join(self.img_path, img_name)
        # 读取对应路径的原图内容，生成图片对象，存储在img中
        img = Image.open(img_item_path)
        torch.random.manual_seed(17)
        img = self.transformimg(img)
        img = np.array(img)

        label_name = self.label_list[item]
        label_item_path = os.path.join(self.label_path, label_name)
        # 读取对应路径的掩码图片内容，生成图片对象，存储在label中
        label = Image.open(label_item_path)
        torch.random.manual_seed(17)
        label = self.transformlab(label)
        #
        label = convert_nonzero_to_one(np.array(label))

        return img, label

    def __len__(self):
        return len(self.img_list)
