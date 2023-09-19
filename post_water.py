from skimage import segmentation, color, measure, morphology
import os
import numpy as np
import cv2 as cv2
import time
import matplotlib.pyplot as plt
import datetime
from post_proc import skeletonize, padding_img, dilate, boundary_proc


def post_overlap_grain(i, boundary_map, xcx_map):
    '''
    Calculate the adjacency relationship of the current precipitation phase
    :param i: Precipitated phase number
    :param boundary_map:
    :param xcx_map:
    :return:
    '''
    tmp = boundary_map[xcx_map == i]
    overlap_grain = np.unique(tmp)
    overlap_grain = np.delete(overlap_grain, np.where(overlap_grain == 0))
    return overlap_grain


def watershed_pic(j, tar_label):
    '''
    Carry out watershed for the given grain j and area tar_label
    :param j:
    :param tar_label:
    :return:
    '''
    # Watershed method, deduct the overlap area, and then perform local watershed on cut_map
    # 1. The target area needs to be deducted first.
    tar_label[tar_label == j] = 0
    tar_label = np.uint8(tar_label)
    # 2. seed  :Corrosion of each grain
    E_kernel = np.ones((5, 5), np.uint8)
    dst = cv2.erode(tar_label, E_kernel)
    sure_fg = np.uint8(dst)
    # 3. boundary，remove small grains that need to be re segmented
    sure_bg = tar_label.copy()
    sure_bg = np.uint8(sure_bg)

    maker = sure_fg
    tar_img = sure_bg
    tar_img = tar_img.astype(np.uint8)

    tar_img = cv2.cvtColor(tar_img, cv2.COLOR_GRAY2RGB)  

    maker = maker.astype('int32')
    maker = cv2.watershed(tar_img, maker)
    tar_img[maker == -1] = [0, 0, 255]
    return maker


def water_grain(i, j, boundary_map, xcx_map, img, thresh_iou):
    '''
    Determine the coverage rate of the grain j covered by the precipitate phase i
    :param i:
    :param j:
    :param boundary_map:
    :param xcx_map:
    :param img:
    :return:
    '''
    tmp_map = np.zeros(boundary_map.shape, dtype=np.uint8)
    tmp_map[boundary_map == j] = j
    # Determine the relationship between precipitates and related grains
    temp_IOU = 0.0
    n_ii = np.count_nonzero(boundary_map[xcx_map == i] == j)
    t_i = np.count_nonzero(boundary_map == j)
    if t_i == 0:
        return None
    temp_IOU += n_ii / t_i  

    if temp_IOU < thresh_iou: 
        return None


    b, a = boundary_map.shape
    cut_map = np.zeros(tmp_map.shape, dtype=np.uint8)
    cut_map[(tmp_map != 0) | (xcx_map == i)] = 255
    cut_map = cut_map.astype(np.uint8)
    x, y, w, h = cv2.boundingRect(cut_map)

    step = 10
    x = x - step if (x - step > 0) else 0
    y = y - step if (y - step > 0) else 0
    w = w + 2 * step if (x + w + 2 * step < a) else (a - x)
    h = h + 2 * step if (y + h + 2 * step < b) else (b - y)

    # Crop the area that needs to be watered
    tar_img = img[y:y + h, x:x + w]
    tar_label = boundary_map[y:y + h, x:x + w]
    tar_label = tar_label.copy()
    tar_xcx = xcx_map[y:y + h, x:x + w]

    flag = 1 # Default mapping required
    label_total = np.unique(tar_label)
    if max(label_total)<=256:
        flag = 0
    else:
        j = j % 256
        # Develop a dict to record the data mapping after processing
        label_dict = {}
        for item in label_total:
            remainder = item % 256
            label_dict[remainder] = item
            # label_dict[item] = remainder
            tar_label[tar_label == item] = remainder

    maker = watershed_pic(j, tar_label)
    maker[maker == -1] = 0
    
    
    if flag == 1:  # mapping
        j = j+256
        for item in np.unique(maker):
            maker[maker == item] = label_dict[item]
    return [maker, x, y]


def water_post(img, boundary, xcx, thresh_iou):  # todo
    '''
    :param img_path:
    :param boundary_path:
    :param xcx_path:
    :return:
    '''
    # xcx = cv2.imread(xcx_path, 0)
    # img = cv2.imread(img_path, 0)
    # boundary = cv2.imread(boundary_path, 0)

    # plt.figure()
    # plt.subplot(131),plt.imshow(img, cmap="gray")
    # plt.subplot(132),plt.imshow(boundary, cmap="gray")
    # plt.subplot(133),plt.imshow(xcx, cmap="gray")
    # plt.show()

    boundary[boundary != 0] = 1

    # To achieve more accurate segmentation, first skeleton the boundary
    skeleton = morphology.skeletonize(boundary).astype(np.uint8) * 255
    # skeleton0 = morphology.skeletonize(boundary)
    # skeleton = skeleton0.astype(np.uint8) * 255
    # boundary = boundary.astype(np.uint8)
    ret, thresh = cv2.threshold(skeleton, 0, 255, cv2.THRESH_BINARY)
    boundary = thresh

    # Because the boundary segmentation is not good, first pad the boundary of the boundary
    boundary = padding_img(~boundary, 0)

    boundary_map, num_boundary = measure.label(boundary, connectivity=1, background=0, return_num=True)
    xcx_map, num_xcx = measure.label(xcx, connectivity=1, background=0, return_num=True)

    for i in range(1, num_xcx + 1):

        overlap_grain = post_overlap_grain(i, boundary_map, xcx_map)

        if len(overlap_grain) == 1:
            continue

        for j in overlap_grain:
            # Calculate and select individual grain regions, 
            # and then perform a watered operation on those that meet the conditions to obtain the processed local slices
            water_info = water_grain(i, j, boundary_map, xcx_map, img, thresh_iou)

            if water_info == None:  # The current j grain is not suitable
                continue

            water_patch, x, y = water_info
            h, w = water_patch.shape
            step = 5
            x1 = x + step
            y1 = y + step
            h1 = h - step
            w1 = w - step

            boundary_map[y1:y + h1, x1:x + w1] = water_patch[step:h1, step:w1] 

    blank_ground = np.zeros(boundary_map.shape, dtype=np.uint8)
    label_b = segmentation.mark_boundaries(blank_ground, boundary_map, (255, 255, 255), background_label=0)

    label_b[label_b > 0] == 1
    label_b = label_b.astype(np.uint8)
    label_b = skeletonize(label_b)

    label_b = cv2.cvtColor(label_b, cv2.COLOR_BGR2GRAY)
    water_img = dilate(label_b, 3)

    return water_img


if __name__ == '__main__':
    today = datetime.date.today()
    thresh_iou = 0.9
    boundary_path = './infer/OM/boundary'
    xcx_path = './infer/OM/xcx'
    img_path =  './data/OM/test_img'
    save_path = './infer/OM/test-water'+ '-' + str(thresh_iou)+"/"
    os.makedirs(save_path, mode=0o777, exist_ok=True) 

    files = os.listdir(xcx_path)
    print(f"Analyzing {len(files)} files")
    print(files)

    for filename in files:
        img_name = filename
        if "checkpoint" in filename:
            continue
        if 'label' in filename:
            a,b = os.path.splitext(filename)
            a = a[:-6]
            img_name = a+b
        print(filename + "---processing")
        pic_img_path = img_path + "/" + img_name
        pic_xcx_path = xcx_path + "/" + filename
        pic_boundary_path = boundary_path + "/" + filename
        pic_save_path = save_path + "/" + filename

        xcx = cv2.imread(pic_xcx_path, 0)
        img = cv2.imread(pic_img_path, 0)
        boundary = cv2.imread(pic_boundary_path, 0)

        water_img = water_post(img, boundary, xcx, thresh_iou)
        
        water_img = boundary_proc(water_img)
        water_img[water_img > 0] = 255

        cv2.imwrite(pic_save_path, water_img)

        plt.figure()
        plt.imshow(water_img)
        plt.show()