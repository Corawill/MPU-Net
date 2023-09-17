import os
import random
import numpy as np
import torch
import math
from pathlib import Path
from skimage import io
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tr
import matplotlib.pyplot as plt
from typing import Callable, Iterable, List, Set, Tuple
from IPython.display import clear_output
from skimage import morphology
import cv2
from PIL import Image
from model.nets.unet import UNet


def setup_seed(seed: int) -> None:
    """
        Set random seed to make experiments repeatable
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # implement same config in cpu and gpu
    torch.backends.cudnn.benchmark = True


def load_img(img_path: Path) -> np.ndarray:
    """
        Load images or npy files, you can add your own type of file in here
    """
    if img_path.suffix == '.npy':
        img = np.load(str(img_path))
    else:
        img = io.imread(str(img_path), 0)
        if len(img.shape) == 3:
            img = np.array(Image.open(img_path).convert("L"))
    return img


class Logger:
    """
        Log and print the information during experiment
    """

    def __init__(self, is_out_log_file=True, file_address=None) -> None:
        self.is_out_log_file = is_out_log_file
        self.file_address = file_address

    def log_print(self, content) -> None:
        print(content)
        if self.is_out_log_file:
            f = open(str(self.file_address), "a")
            f.write(content)
            f.write("\n")
            f.close()


def adjust_learning_rate(optimizer, epoch, learning_rate, dacay_ratio: float = 0.8, decay_num: int = 10,
                         lower_limit: float = 1e-6) -> None:
    """
        Sets the learning rate to the initial LR decayed by dacay_ratio every decay_num epochs until the lower_limit
    """
    lr = learning_rate * (dacay_ratio ** (epoch // decay_num))
    if not lr < lower_limit:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def check_result(epoch, img, label, xcx, output, xcx_output, weight, xcx_weight, checkpoint_path, logger, description="val_"): # check the output of dataset
    """
        Check visualization of result during training
    """
    logger.log_print("Image size is {},    min is {}, max is {}".format(img.shape, np.amin(img), np.amax(img)))
    logger.log_print("Label size is {},    min is {}, max is {}".format(label.shape, np.amin(label), np.amax(label)))
    logger.log_print("Output size is {},   min is {}, max is {}".format(output.shape, np.amin(output), np.amax(output)))
    logger.log_print("Weight-0 size is {}, min is {}, max is {}".format(weight[:, :, 0].shape, np.amin(weight[:, :, 0]), np.amax(weight[:, :, 0])))
    logger.log_print("Weight-1 size is {}, min is {}, max is {}".format(weight[:, :, 1].shape, np.amin(weight[:, :, 1]), np.amax(weight[:, :, 1])))
    plt.figure(figsize=(20, 40))
    plt.subplot(1, 9, 1), plt.imshow(img,    cmap="gray"), plt.title("img"), plt.axis("off")
    plt.subplot(1, 9, 2), plt.imshow(label,  cmap="gray"), plt.title("label"), plt.axis("off")
    plt.subplot(1, 9, 3), plt.imshow(xcx,  cmap="gray"), plt.title("label-xcx"), plt.axis("off")
    plt.subplot(1, 9, 4), plt.imshow(output, cmap="gray"), plt.title("output"), plt.axis("off")
    plt.subplot(1, 9, 5), plt.imshow(xcx_output,  cmap="gray"), plt.title("output-xcx"), plt.axis("off")
    plt.subplot(1, 9, 6), plt.imshow(weight[:, :, 0], cmap="plasma"), plt.title("weight-0"), plt.axis("off")
    plt.subplot(1, 9, 7), plt.imshow(weight[:, :, 1], cmap="plasma"), plt.title("weight-1"), plt.axis("off")
    plt.subplot(1, 9, 8), plt.imshow(xcx_weight[:, :, 0], cmap="plasma"), plt.title("xcx-weight-1"), plt.axis("off")
    plt.subplot(1, 9, 9), plt.imshow(xcx_weight[:, :, 1], cmap="plasma"), plt.title("xcx-weight-1"), plt.axis("off")
    plt.savefig(str(Path(checkpoint_path, description + str(epoch).zfill(3) + "_result.png")))
    plt.show()




def plot(epoch, train_value_list, val_value_list, checkpoint_path, find_min_value=True, curve_name="loss",
         ignore_epoch: int = 2) -> None:
    """
        Plot training cure during training, and draw target epoch in the figure
    """
    clear_output(True)
    plt.figure()
    target_value = 0
    target_func = 'None'
    if find_min_value and len(val_value_list) > ignore_epoch:
        target_value = min(val_value_list[ignore_epoch:])
        target_func = 'min'
    elif find_min_value is False and len(val_value_list) > ignore_epoch:
        target_value = max(val_value_list[ignore_epoch:])
        target_func = 'max'
    title_name = 'Epoch {}. train ' + curve_name + ': {:.4f}. val ' + curve_name + ': {:.4f}. ' + ' val' + target_func + ' ' + curve_name + ': {:.4f}. '
    plt.title(title_name.format(epoch, train_value_list[-1], val_value_list[-1], target_value))
    plt.plot(train_value_list, color="r", label="train " + curve_name)
    plt.plot(val_value_list, color="b", label="val " + curve_name)
    if len(val_value_list) > ignore_epoch:
        plt.axvline(x=val_value_list.index(target_value), ls='-', c='green')
        plt.legend(loc='best')
    plt.savefig(str(Path(checkpoint_path, curve_name + '_curve.png')))
    plt.show()


def uniq(a: Tensor) -> Set:
    """ Get the set of unique value in tensor """
    return set(torch.unique(a.cpu()).numpy())


def is_sset(a: Tensor, sub: Iterable) -> bool:
    """ 
        Judge the unique value of a is the subset of sub 
        is_set(a, [0,1])
    """
    return uniq(a).issubset(sub)


class GpuDetector:
    """
        Gpu detector to address the problems related to the gpu.
    """

    def __init__(self):
        self._gpu_number = torch.cuda.device_count()  # if there is np gpu, return 0
        self._gpu_idx_in_use = -1  # which gpu to use
        self._gpu_is_available = False
        for gpu_idx in range(self._gpu_number):
            try:
                self.__check_gpu_status(gpu_idx)
            except AssertionError as e:
                pass
            else:
                self._gpu_is_available = True
                self._gpu_idx_in_use = gpu_idx
                break

    def set_gpu_device(self, gpu_idx: int):
        """
            Setting certain gpu advice to use
        """
        try:
            self.__check_gpu_status(gpu_idx)
        except AssertionError as e:  # if there is error, raise it to upper call
            raise e
        else:
            self._gpu_is_available = True
            self._gpu_idx_in_use = gpu_idx

    def get_current_gpu(self) -> Tuple:
        """
            Get current gpu status and gpu device
        """
        return False, self._gpu_idx_in_use

    def __check_gpu_status(self, gpu_idx: int):
        """
            Check gpu status, some errors will return by AssertionError
        """
        assert torch.cuda.is_available(), "This computer has no gpu"
        assert isinstance(gpu_idx, int) and gpu_idx >= 0, "The gpu index must be int and greater than 0"
        assert gpu_idx <= (self._gpu_number - 1), "There is only {} gpus and has no gpu with idx {}".format(
            self._gpu_number, gpu_idx)
        assert torch.cuda.get_device_capability(gpu_idx)[
                   0] >= 2.0, "The device capability of gpu {} with name {} is lower than 2.0".format(gpu_idx,
                                                                                                      torch.cuda.get_device_name(
                                                                                                          gpu_idx))