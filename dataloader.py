import os, time, random
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from pathlib import Path
from skimage.measure import label, regionprops
import scipy.ndimage as ndimage
from skimage import morphology
import torchvision.transforms as tr
from utils import *
from typing import List, Tuple, Dict

# 把原图根据cropsize的设定进行裁剪
class SegDataset(Dataset):
    """
        Segmentation Dataset, is suit to 2d and 3d images
    """
    def __init__(self, imgs_dir: Path, data_names: List, weight_dir=None, crop_size=None, use_augment=[False, False, False], depth=None, norm_transform=None) -> None:
        self._img_path = []
        self._label_path = []
        self._xcx_path = []
        self._weight_path = []
        self._xcx_weight_path = []
        xcx_weight_dir = Path(imgs_dir, 'xcx_uw')
        # 读取数据
        for item in data_names:
            self._img_path.append(Path(imgs_dir, 'images', item))
            self._label_path.append(Path(imgs_dir, 'labels', item))
            self._xcx_path.append(Path(imgs_dir, 'labels_xcx', item))
            if weight_dir is not None:
                self._weight_path.append(Path(weight_dir, item.split('.')[0] + '.npy'))
            if xcx_weight_dir is not None:
                self._xcx_weight_path.append(Path(xcx_weight_dir, item.split('.')[0] + '.npy'))
        # 进行数据增强，来扩充数据
        self._augmentor = DataAugmentor(crop_size, use_augment, depth)
        # 为了配适3D结构的图像
        self._depth = depth    # if the images is 3D structure,  this param recode the depth of the 3D images
        self._norm_transform = norm_transform
        
        img = load_img(self._img_path[0])
        label = load_img(self._label_path[0])
        xcx = load_img(self._xcx_path[0])

        self._h, self._w = img.shape[:2]
        self._class_num = np.amax(label) + 1
    
    def __getitem__(self, index: int) -> Dict:

        # read images
        if self._depth is None:    # 2D analysis，直接读入
            img    = load_img(self._img_path[index])  
            label  = load_img(self._label_path[index])
            xcx    = load_img(self._xcx_path[index])  
            weight = load_img(self._weight_path[index])
            xcx_weight = load_img(self._xcx_weight_path[index])

        else:                      # 3D analysis
            img    = np.zeros((self._h, self._w, self._depth))
            label  = np.zeros((self._h, self._w, self._depth))
            weight = np.zeros((self._h, self._w, self._depth, self._class_num))
            # 没太懂这部分的内容
            for idx, depth_idx in enumerate(range(index, index + self._depth)):
                if depth_idx > len(self._img_path):   # it will copy the last image to form 3D image
                    img[: , :, idx]       = load_img(self._img_path[len(self._img_path) - 1])
                    label[: , :, idx]     = load_img(self._label_path[len(self._img_path) - 1])
                    weight[: , :, idx, :] = load_img(self._weight_path[len(self._img_path) - 1])
                else:
                    img[: , :, idx]       = load_img(self._img_path[depth_idx])
                    label[: , :, idx]     = load_img(self._label_path[depth_idx])
                    weight[: , :, idx, :] = load_img(self._weight_path[depth_idx])
        
        # data augmentation 数据增强的操作
        if len(self._weight_path) != 0:
            in_transfer = [img, label, weight, xcx, xcx_weight]
            data = self._augmentor.start_augmentation(in_transfer)
            img, label, weight, xcx, xcx_weight = data[0], data[1], data[2], data[3], data[4]
            
        else: 
            in_transfer = [img, label, xcx]
            data = self._augmentor.start_augmentation(in_transfer)
            img, label, xcx = data[0], data[1], data[2]
        
        # to tensor
        if self._depth is None:              # 2D analysis->[C,H,W]
            # print("err-------------------------------------------")
            # print(img.shape)
            # print(self._norm_transform)
            img = self._norm_transform(img)
            label = torch.from_numpy(label[np.newaxis, :, :])
            xcx = torch.from_numpy(xcx[np.newaxis, :, :])
            if len(self._weight_path) != 0:
                weight = torch.from_numpy(weight.transpose((2, 0, 1)))
            if len(self._xcx_weight_path) != 0:
                xcx_weight = torch.from_numpy(xcx_weight.transpose((2, 0, 1)))

        else:                                # 3D analysis->[C,D,H,W]
            img    = np.ascontiguousarray(img, dtype=np.float32)
            label  = np.ascontiguousarray(label, dtype=np.float32)
            img = self._norm_transform(img)
            img = img.unsqueeze(0)
            label = torch.from_numpy(label.transpose((2, 0, 1))[np.newaxis, :, :, :])
            if len(self._weight_path) != 0:
                weight = np.ascontiguousarray(weight, dtype=np.float32)
                weight = torch.from_numpy(weight.transpose((3, 2, 0, 1))) # shape(H, W, D, C) -> (C, D, H, W)
        
        out_data = {'img': img, 'label': label, 'xcx':xcx}
        if len(self._weight_path) != 0:
            out_data['weight'] = weight
        if len(self._xcx_weight_path) != 0:
            out_data['xcx_weight'] = xcx_weight

        return out_data        
    
    def __len__(self) -> int:
        return len(self._img_path)

    # 数据增强

class DataAugmentor:
    """
        The class of data augmentor, you can add your own function in this class
    """  
    def __init__(self, crop_size:int = None, use_augment: List = [False, False, False], depth = None) -> None:
        self.__crop_size   = crop_size
        self.__use_augment = use_augment
        self.__depth = depth

    def start_augmentation(self, data) -> List:
        aug_num = len(self.__use_augment)
        # print('图像数目', len(data))
        # print('原图大小', data[0].size())
        # 进行不同类型的数据增强
        for aug_idx in range(aug_num):
            if aug_idx == 0 and self.__use_augment[aug_idx]: 
                data = self.__rand_rotation(data)               # 0, 90, 180, 270，随机旋转
            elif aug_idx == 1 and self.__use_augment[aug_idx]:
                data = self.__rand_vertical_flip(data)          # p<0.5, flip
            elif aug_idx == 2 and self.__use_augment[aug_idx]:
                data = self.__rand_horizontal_flip(data)        # p<0.5, flip
            elif aug_idx == 3 and self.__use_augment[aug_idx] and self.__depth is not None:
                data = self.__rand_z_filp(data)                 # p<0.5, flip
        data = self.__rand_crop(data)

        return data
 
    def __rand_rotation(self, data) -> List:
        angle = random.choice([0, 90, 180, 270])# 随机旋转
        rotate_idx = angle / 90
        data_num = len(data)
        for data_idx in range(data_num):
            data[data_idx] = np.rot90(data[data_idx], rotate_idx).copy()
        return data

    def __rand_vertical_flip(self, data) -> List:
        p = random.random()
        if p < 0.5:
            data_num = len(data)
            for data_idx in range(data_num):
                data[data_idx] = np.flipud(data[data_idx]).copy()
        return data

    def __rand_horizontal_flip(self, data) -> List:
        p = random.random()
        if p < 0.5:
            data_num = len(data)
            for data_idx in range(data_num):
                data[data_idx] = np.fliplr(data[data_idx]).copy()
        return data

    def __rand_z_filp(self, data) -> List:
        p = random.random()
        if p < 0.5:
            data_num = len(data)
            for data_idx in range(data_num):
                data[data_idx] = np.flip(data[data_idx], 2).copy()
        return data

    def __rand_crop(self, data) -> List:
        random_h = random.randint(0, data[0].shape[0] - self.__crop_size)
        random_w = random.randint(0, data[0].shape[1] - self.__crop_size)
        data_num = len(data)
        for data_idx in range(data_num):
            data[data_idx] = data[data_idx][random_h: random_h + self.__crop_size, random_w: random_w + self.__crop_size]    
            # print(data[data_idx].size())
        return data
