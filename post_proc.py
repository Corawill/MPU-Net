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
    Read grayscale image and convert to binary value 0,1
    :param image_path:
    :return: 0,1 binary matrix
    """
    if Path(image_path).exists():
        image = Image.open(image_path).convert('L')
        image = np.clip(image, 0, 1)
        image = np.array(image)
        return image
    else:
        return None

        
def padding_img(img, element, pad_size = 5):
    '''
    :param img:
    :return:
    '''
    a, b = img.shape
    img[0:pad_size, 0:b] = element
    img[0:a, 0:pad_size] = element
    img[a - pad_size + 1:a, 0:b] = element
    img[0:a, b - pad_size + 1:b] = element

    return img


def dilate(image, kernel_size):
    """
    dilate
    :param image:
    :param kernel_size:
    :return:
    """
    kernel = skimage.morphology.disk(kernel_size)
    img_dialtion = skimage.morphology.dilation(image, kernel)
    return img_dialtion

def erosion(image, kernel_size):
    """
    :param image:
    :param kernel_size:
    :return:
    """
    kernel = skimage.morphology.disk(kernel_size)
    img_erosion = skimage.morphology.erosion(image, kernel)
    return img_erosion


def skeletonize(image):
    """
    :param image:
    :return:
    """
    image = morphology.skeletonize(image)
    image = np.where(image > 0, 1, 0).astype(np.uint8)
    return image


def remove_small_objects(image, min_size):
    """
    :param image:
    :param min_size:
    :return:
    """
    image = np.array(image, dtype=bool)
    image = morphology.remove_small_objects(image, min_size=min_size, connectivity=2)
    image = np.where(image > 0, 1, 0).astype(np.uint8)
    return image


def prun(image, kernel_size):
    label_map, num_label = measure.label(image, connectivity=1, background=1, return_num=True)
    result = np.zeros(label_map.shape, dtype=np.uint8)  
    D_kernel = np.ones((kernel_size, kernel_size), np.uint8)

    for i in range(1, num_label + 1):
        tmp = np.zeros(label_map.shape, dtype=np.uint8)  
        tmp[label_map == i] = 1

        dil = cv2.dilate(tmp, D_kernel)
        dst = cv2.erode(dil, D_kernel)

        result[dst == 1] = 255

    result = 255 - result
    result[result == 255] = 1
    result = np.uint8(result)
    return result



def prun_xcx(image, kernel_size):
    """
    Remove small voids inside the precipitated phase
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
    return result


def show(image):
    image_show = image.copy()
    image_show[image_show == 1] = 255
    cv2.imshow('image', image_show)
    cv2.waitKey()



def boundary_proc(image):
    """
    Grain boundary post-processing operations
    :param image: 
    :return:
    """
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
    Precipitated phase post-processing operations
    :param image: 
    :return: 
    """
    image = remove_small_objects(image, 50)
    image = erosion(image, 3)
    image = dilate(image, 3)
    image = prun_xcx(image, 8)
    image = remove_small_objects(image, 50)
    return image


if __name__ == '__main__':
    proc_method = 'xcx'
    boundary_path = './infer/OM/xcx'
    save_path = './post/OM/xcx'
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

        pred[pred == 1] = 255
        Image.fromarray(pred).save(str(save_path.joinpath(image_path.name)))