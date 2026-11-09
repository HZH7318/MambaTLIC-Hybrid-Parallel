import os
import numpy as np
import torch
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        :param root_dir: 根目录，包含训练集和测试集文件夹
        :param split: 数据集的划分，如 'train' 或 'test'
        :param transform: 数据转换操作，默认是None
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.npy_folder = os.path.join(root_dir, split)
        # print(self.npy_folder)
        self.npy_list = [f for f in os.listdir(self.npy_folder) if f.endswith('.npy')]

    def __len__(self):
        """
        返回数据集的长度（即.npy文件的数量）
        """
        return len(self.npy_list)

    def __getitem__(self, idx):
        """
        :param idx: 索引，获取数据集中的第idx个数据
        :return: 处理后的数据（tensor）和文件名
        """
        npy_path = os.path.join(self.npy_folder, self.npy_list[idx])
        
        # 加载 .npy 数据，假设每个.npy文件是一个 (7, 256, 256) 的数组
        npy_data = np.load(npy_path)
        
        # 转换为torch tensor，默认是float32类型
        npy_data = npy_data.astype(np.float32) 
        image = torch.tensor(npy_data, dtype=torch.float32)
        
        # 如果有转换操作，应用转换
        if self.transform:
            image = self.transform(image)

        return image, self.npy_list[idx]  # 返回数据和文件名

class ResBaseDataset(Dataset):
    def __init__(self, root_dir):
       
        self.files = sorted([os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.pt')])
    def __len__(self):
   
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
        return  data['name'], data['imput_normal'], data['res'],data['x_hat']
