from skimage import io
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from utils import *
from model.losses.bcw_loss import *
from model.losses.uw_loss import *

# The first step of deep learning experiment is data preprocessing.   

def data_preprocess(dataset_name: str, in_path: Path, out_path: Path, zfill_num: int = 3, suffix: str = "png") -> None:
    if dataset_name == "isbi2012":
        print("isbi2012")
    else:
        return

# 把数据集加载进来

def prepare_weight_map(labels_path: Path, out_path: Path, func_names: List, class_num: int = 2) -> None:
    """
        Generate the weight map in data directory, such as balanced class weight or unet weight
    """
    file_paths = []
    for file in labels_path.rglob("*.png"):
        if Path.is_file(file) and ".ipynb_checkpoints" not in str(str(file)):
            file_paths.append(file)
    file_paths.sort()
    # init
    bcw_weight = BalancedClassWeight(class_num)
    uw_weight = UnetWeight()
    print("Weight Map Calculation...")
    for file_path in file_paths:
        file_name = file_path.name
        print("Analysing {} ..." .format(file_name))
        label = load_img(file_path)
        for func_name in func_names:
            if 'xcx' in labels_path.name and func_name == 'uw':
                weight_path = Path(out_path, 'xcx_uw')
            else:
                weight_path = Path(out_path, func_name)
            weight_path.mkdir(parents=True, exist_ok=True)
            weight = np.zeros((label.shape[0], label.shape[1], class_num))
            if func_name == 'bcw':
                weight = bcw_weight.get_weight(label)
            # 在边界粘连项目会用到
            elif func_name == 'uw':
                assert class_num == 2, "The unet weight only support binary segmentation"
                weight = uw_weight.get_weight(label)

            print("{} for {} has done".format(func_name, file_name))
            print("Visualization:")
            for vis_idx in range(class_num):
                print("{} {}, the max value is {} and the min value is {}".format(func_name, vis_idx, 
                                                                                  np.amax(weight[:, :, vis_idx]),
                                                                                  np.amin(weight[:, :, vis_idx])))
                # print('    and the unique values are {}'.format(np.unique(weight[:, :, vis_idx])))
            plt.figure(figsize=(10, 10))
            plt.subplot(1, class_num + 1, 1), plt.imshow(label, cmap="gray"), plt.title("label"), plt.axis("off")
            for vis_idx in range(class_num):
                plt.subplot(1, class_num + 1, vis_idx + 2)
                plt.imshow(weight[:, :, vis_idx]), plt.title("{} {}".format(func_name, vis_idx)), plt.axis("off")
            plt.show()
            # print(str(Path(weight_path, file_name.replace('png', 'npy'))))
            np.save(str(Path(weight_path, file_name.replace("png", "npy"))), weight)
