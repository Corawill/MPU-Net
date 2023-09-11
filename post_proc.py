import time
import skimage.io
import numpy as np
import cv2 as cv2
from PIL import Image
from pathlib import Path
from skimage import morphology
from skimage import measure
import matplotlib.pyplot as plt

def read_image(image_path):
    """
    读灰度图像，并转换为0,1二值
    :param image_path:
    :return: 0,1二值矩阵
    """
    if Path(image_path).exists():
        # 读取灰度图像
        image = Image.open(image_path).convert('L')
        image = np.clip(image, 0, 1)
        image = np.array(image)
        skimageshow(image)
        return image
    else:
        return None


def save_image(image, save_path):
    """
    0,1 二值矩阵转化为灰度图存储
    :param image:
    :param save_path:
    :return:
    """
    image[image == 1] = 255
    Image.fromarray(image).save(str(save_path))


        
def padding_img(img, element, pad_size = 5):
    '''
    边界分割不好，把边界打包一下，注意padding的语言是什么
    :param img:
    :return:
    '''
    a, b = img.shape
    # pad_size = 5
    img[0:pad_size, 0:b] = element
    img[0:a, 0:pad_size] = element
    img[a - pad_size + 1:a, 0:b] = element
    img[0:a, b - pad_size + 1:b] = element

    return img


def dilate(image, kernel_size):
    """
    膨胀操作
    :param image:
    :param kernel_size:
    :return:
    """
    kernel = skimage.morphology.disk(kernel_size)
    img_dialtion = skimage.morphology.dilation(image, kernel)
    skimageshow(img_dialtion)
    return img_dialtion

def erosion(image, kernel_size):
    """
    腐蚀操作
    :param image:
    :param kernel_size:
    :return:
    """
    kernel = skimage.morphology.disk(kernel_size)
    img_erosion = skimage.morphology.erosion(image, kernel)
    skimageshow(img_erosion)
    return img_erosion


def skeletonize(image):
    """
    骨架化
    :param image:
    :return:
    """
    image = morphology.skeletonize(image)
    image = np.where(image > 0, 1, 0).astype(np.uint8)
    skimageshow(image)
    return image


def remove_small_objects(image, min_size):
    """
    去除小目标
    :param image:
    :param min_size:
    :return:
    """
    image = np.array(image, dtype=bool)
    image = morphology.remove_small_objects(image, min_size=min_size, connectivity=2)
    image = np.where(image > 0, 1, 0).astype(np.uint8)
    skimageshow(image)
    return image


def prun(image, kernel_size):
    label_map, num_label = measure.label(image, connectivity=1, background=1, return_num=True)
    result = np.zeros(label_map.shape, dtype=np.uint8)  # 初始化结果数组
    D_kernel = np.ones((kernel_size, kernel_size), np.uint8)

    for i in range(1, num_label + 1):
        tmp = np.zeros(label_map.shape, dtype=np.uint8)  # 初始化临时数组
        tmp[label_map == i] = 1

        # 应用膨胀和腐蚀操作到临时数组
        dil = cv2.dilate(tmp, D_kernel)
        dst = cv2.erode(dil, D_kernel)

        # 将当前标签区域的结果添加到最终结果中 
        result[dst == 1] = 255

    # 进行后续处理
    result = 255 - result
    result[result == 255] = 1
    result = np.uint8(result)
    # skimageshow(result)
    return result


# def prun(image, kernel_size):
#     """
#     去除小枝杈
#     晶粒为0
#     晶界为1
#     """
#     label_map, num_label = measure.label(image, connectivity=1, background=1, return_num=True)
#     result = np.zeros(label_map.shape)
#     D_kernel = np.ones((kernel_size, kernel_size), np.uint8)
#     for i in range(1, num_label + 1):
#         tmp = np.zeros(label_map.shape)
#         tmp[label_map == i] = 1
        
#         dil = cv2.dilate(tmp, D_kernel)
#         dst = cv2.erode(dil, D_kernel)
#         result[dst == 1] = 255
#     result = 255 - result
#     result[result == 255] = 1
#     result = np.uint8(result)
#     skimageshow(result)
#     return result


def prun_xcx(image, kernel_size):
    """
    去除析出相内部的小空隙
    背景为0
    析出相为1
    """
    label_map, num_label = measure.label(image, connectivity=1, background=0, return_num=True)
    result = np.zeros(label_map.shape, dtype=np.uint8)
    for i in range(1, num_label + 1):
        tmp = np.zeros(label_map.shape, dtype=np.uint8)
        tmp[label_map == i] = 1
        D_kernel = np.ones((kernel_size, kernel_size), np.uint8)  # 小图像
        dil = cv2.dilate(tmp, D_kernel)
        dst = cv2.erode(dil, D_kernel)
        result[dst == 1] = 255
    result[result == 255] = 1
    result = np.uint8(result)
    skimageshow(result)
    return result


def show(image):
    image_show = image.copy()
    image_show[image_show == 1] = 255
    cv2.imshow('image', image_show)
    cv2.waitKey()


def skimageshow(image):
    if False:
        skimage.io.imshow(image)
        skimage.io.show()


def boundary_proc(image):
    """
    晶界后处理操作
    :param image: 要求为0,1二值图，1为晶界
    :return: 0,1二值图
    """
    # 晶界后处理
    image = dilate(image, 8)
    image = skeletonize(image)
    image = remove_small_objects(image, 20)
    image = dilate(image, 3)
    image = padding_img(image, element=1, pad_size=8)
    t1 = time.time()
    image = prun(image, 8)
    t2 = time.time()
    print(t2-t1)
    image = remove_small_objects(image, 100)
    return image


def xcx_proc(image):
    """
    析出相后处理操作
    :param image: 要求为0,1二值图，1为析出相
    :return: 0,1二值图
    """
    # 析出相后处理方式
    image = remove_small_objects(image, 50)
    image = erosion(image, 3)
    image = dilate(image, 3)
    image = prun_xcx(image, 8)
    image = remove_small_objects(image, 50)
    return image


if __name__ == '__main__':
    # folder_name = "gyy-split-water"
    # boundary_path = '/root/data/ImgSegmentation-old/data/zxy-yw2021/gangyanyuan-state1/split_test_little/'+folder_name
    # save_path = '/root/data/ImgSegmentation-old/data/zxy-yw2021/gangyanyuan-state1/split_test_little-samewidth/'+folder_name
    #
    proc_method = 'xcx'
    boundary_path = '/root/data/zhangxinyi/data/gyy-state2-train/infer/OM/xcx'
    save_path = '/root/data/zhangxinyi/data/gyy-state2-train/post/OM/xcx'
    root_path = Path(boundary_path)
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    for image_path in root_path.iterdir():
        if 'checkpoints' in str(image_path):
            continue
        image = read_image(image_path)

        if proc_method == 'boundary':
            pred = boundary_proc(image)
        elif proc_method == 'xcx':
            pred = xcx_proc(image)
        else:
            raise RuntimeError('proc_method must be boundary or xcx!')

        print(image_path.name)

        plt.figure("final image")
        plt.imshow(pred)
        plt.show()

        save_image(pred, str(save_path.joinpath(image_path.name)))