#%%
"""
@File     : post_water.py
@Project  : gyy-water
@Time     : 2023.1.10 19:58
@Author   : Cora
@Contact_1: 2641014054@qq.com
@Software : PyCharm
@Script   : 分水岭后处理方法的主要部分
"""

from skimage import data, io, segmentation, color, measure, filters, morphology
from skimage.measure import regionprops
from skimage.future import graph
import os
from pathlib import Path
import numpy as np
import cv2 as cv2
import scipy
import time
import matplotlib.pyplot as plt
import datetime
from post_proc import read_image, erosion, skeletonize, padding_img, \
    save_image, dilate, skimageshow, boundary_proc

def label_img(img):
    '''
    给每个晶粒打标签
    :param img:
    :return: 标记后的数据和label值
    '''
    img_map, num_label = measure.label(img, connectivity=1, background=0, return_num=True)
    color_map = color.label2rgb(img_map)
    # print the map
    # plt.figure()
    # plt.imshow(color_map)
    # plt.show()
    return img_map, num_label

def post_overlap_grain(i, boundary_map, xcx_map):
    '''
    计算当前析出相的邻接关系
    :param i: 析出相编号
    :param boundary_map:
    :param xcx_map:
    :return:
    '''
    # tmp = np.zeros(boundary_map.shape)
    tmp = boundary_map[xcx_map == i]
    # tmp[xcx_map != i] = 0
    # 与析出相 相关的晶粒
    overlap_grain = np.unique(tmp)
    # print("overlap 计算", overlap_grain)
    # 去掉背景0
    overlap_grain = np.delete(overlap_grain, np.where(overlap_grain == 0))
    return overlap_grain


def watershed_pic(j, tar_label):
    '''
    针对既定晶粒j和区域tar_label进行分水岭
    :param j:
    :param tar_label:
    :return:
    '''
    # 分水岭方法，将overlap的区域扣掉，然后对cut_map进行局部分水岭
    # 1. 需要先将目标区域扣除。
    tar_label[tar_label == j] = 0
    tar_label = np.uint8(tar_label)
    # 2. seed  :对各个晶粒进行腐蚀
    E_kernel = np.ones((5, 5), np.uint8)
    dst = cv2.erode(tar_label, E_kernel)
    sure_fg = np.uint8(dst)
    # 3. boundary，去掉需要重新分割的小晶粒
    sure_bg = tar_label.copy()
    sure_bg = np.uint8(sure_bg)

    # print("tar_label不应该改变", tar_label.shape)
    # print(np.unique(tar_label))

    maker = sure_fg
    # 因为边界扭曲，重新调整tar_img，之前是原图
    tar_img = sure_bg
    tar_img = tar_img.astype(np.uint8)

    # 因为经常出现empty的问题，所以img存一下看看
    # cv2.imwrite(str(i)+".png", img)
    tar_img = cv2.cvtColor(tar_img, cv2.COLOR_GRAY2RGB)  # 三通道图像才满足watershed

    # print("tar_img shape:", tar_img.shape)
    maker = maker.astype('int32')
    maker = cv2.watershed(tar_img, maker)
    tar_img[maker == -1] = [0, 0, 255]

    # print("tar_label不应该改变", maker.shape)
    # print(np.unique(maker))

    # plt.figure(figsize=(10, 5))
    # # 分水岭相关参数展示
    # plt.subplot(1, 4, 1), plt.title('sure_fg')
    # plt.imshow(sure_fg), plt.axis('off')

    # plt.subplot(1, 4, 2), plt.title('sure_bg')
    # plt.imshow(sure_bg), plt.axis('off')

    # plt.subplot(1, 4, 3), plt.title('maker')
    # plt.imshow(maker), plt.axis('off')

    # plt.subplot(1, 4, 4), plt.title('result')
    # plt.imshow(tar_img), plt.axis('off')

    return maker


def water_grain(i, j, boundary_map, xcx_map, img, thresh_iou):
    '''
    对析出相i覆盖的晶粒j判断覆盖率，
    :param i:
    :param j:
    :param boundary_map:
    :param xcx_map:
    :param img:
    :return:
    '''
    tmp_map = np.zeros(boundary_map.shape, dtype=np.uint8)
    tmp_map[boundary_map == j] = j
    # plt.imshow(tmp_map)
    # plt.show()
    # 判断析出相和相关晶粒之间的IOU关系
    temp_IOU = 0.0
    n_ii = np.count_nonzero(boundary_map[xcx_map == i] == j)
    t_i = np.count_nonzero(boundary_map == j)
    if t_i == 0:
        return None
    temp_IOU += n_ii / t_i  # 改进IOU = A∩B/A，A为晶粒，表示晶粒被析出相遮挡
    # IOU<0.5 不进行处理
    if temp_IOU < thresh_iou:  # 可以调整的超参数
        # print("析出相:", i, "晶粒:", j, "IOU值:", temp_IOU)
        return None
    # else:
        # print("---------------------析出相:", i, "晶粒:", j, "IOU值:", temp_IOU)

    # 裁剪出对应区域的晶粒部分
    b, a = boundary_map.shape
    cut_map = np.zeros(tmp_map.shape, dtype=np.uint8)
    cut_map[(tmp_map != 0) | (xcx_map == i)] = 255
    cut_map = cut_map.astype(np.uint8)
    x, y, w, h = cv2.boundingRect(cut_map)
    # print(x,y,w,h)
    # print(a,b)
    # 重新调整x,y,w,h的定位，扩大满足区域面积
    step = 10
    x = x - step if (x - step > 0) else 0
    y = y - step if (y - step > 0) else 0
    w = w + 2 * step if (x + w + 2 * step < a) else (a - x)
    h = h + 2 * step if (y + h + 2 * step < b) else (b - y)

    # 对需要watershed的区域进行裁剪
    # print("左上角坐标x,y:", x, y, "宽高w,h:", w, h)
    tar_img = img[y:y + h, x:x + w]
    # tar_img = tar_img.copy()
    tar_label = boundary_map[y:y + h, x:x + w]
    tar_label = tar_label.copy()
    tar_xcx = xcx_map[y:y + h, x:x + w]
    # tar_xcx = tar_xcx.copy()

    # print('裁剪图像包含区域：', np.unique(tar_label))
    # 将要被watershed分割的区域
    # plt.figure(figsize=(10, 5))

    # plt.subplot(1, 4, 1), plt.title('boundary')
    # plt.imshow(tar_label), plt.axis('off')

    # plt.subplot(1, 4, 2), plt.title('xcx')
    # plt.imshow(tar_xcx), plt.axis('off')

    # plt.subplot(1, 4, 3), plt.title('img')
    # plt.imshow(tar_img), plt.axis('off')

    # print("分水岭前——water_grain:tar_label:", np.unique(tar_label))
    flag = 1 # 默认需要映射
    # tar_label的label范围处理
    label_total = np.unique(tar_label)
    if max(label_total)<=256:
        flag = 0
    else:
        j = j%256
        # 制定一个dict，记录处理之后的数据mapping
        label_dict = {}
        for item in label_total:
            remainder = item % 256
            label_dict[remainder] = item
            # label_dict[item] = remainder
            tar_label[tar_label == item] = remainder
        # label_magnification = label_total/256
        # label_remainder = label_total%256
        # print("label_total        :",label_total)
        # print("label_magnification:",label_magnification)
        # print("label_remainder    :",label_remainder)
        
    # save_color_boundary = color.label2rgb(tar_label)  # 根据不同的标记显示不同的颜色
    # io.imsave("E:\\1_Study\\Graduate_Project\\Coding\\gyy-water\\data\\step\\" + str(00) + "-" + str(00) + "tar_label.png", save_color_boundary)
    # 分水岭方法
    maker = watershed_pic(j, tar_label)
    maker[maker == -1] = 0
    
    
    # 对分水岭之后的结果重新处理,label mapping，注意这个时候j已经没有了
    if flag == 1:  # mapping操作
        j = j+256
        for item in np.unique(maker):
            maker[maker == item] = label_dict[item]

    print("分水岭后——water_grain:tar_label:", np.unique(maker))
    # label_total.remove(j)
    # aim_index = np.where(label_total==j)[0][0]
    # np.delete(label_total,np.where(label_total==j)[0][0])
    # if label_total.all() == np.unique(maker).all():
    #     print("label前后match！！-----------------------------------------------------------")
    
    # save_color_boundary = color.label2rgb(maker)  # 根据不同的标记显示不同的颜色
    # io.imsave("E:\\1_Study\\Graduate_Project\\Coding\\gyy-water\\data\\step\\" + str(00) + "-" + str(00) + ".png", save_color_boundary)
    return [maker, x, y]


def water_post(img_path, boundary_path, xcx_path, thresh_iou):  # todo
    '''
    watershed后处理图像完整流程
    :param img_path:
    :param boundary_path:
    :param xcx_path:
    :return:
    '''
    # 首先读取图像(图像内容0-1化) img, boundary,xcx
    xcx = cv2.imread(xcx_path, 0)
    img = cv2.imread(img_path, 0)
    boundary = cv2.imread(boundary_path, 0)

    plt.figure()
    plt.subplot(131),plt.imshow(img)
    plt.subplot(132),plt.imshow(boundary)
    plt.subplot(133),plt.imshow(xcx)
    plt.show()

    boundary[boundary != 0] = 1

    # 为了分割更准确，先将boundary骨架化
    skeleton0 = morphology.skeletonize(boundary)
    skeleton = skeleton0.astype(np.uint8) * 255
    boundary = boundary.astype(np.uint8)
    ret, thresh = cv2.threshold(skeleton, 0, 255, cv2.THRESH_BINARY)
    boundary = thresh

    # 因为边界分割不好，所以先把boundary的边界padding一下
    boundary = padding_img(~boundary, 0)

    # 把各个晶粒都label一下
    boundary_map, num_boundary = label_img(boundary)
    xcx_map, num_xcx = label_img(xcx)

    # 开始逐个晶粒的处理
    for i in range(1, num_xcx + 1):
        # print(i, "/", num_xcx)
        # 计算当前析出相和谁内接
        overlap_grain = post_overlap_grain(i, boundary_map, xcx_map)
        # 如果只和一个晶粒内接的话，则处在某个晶粒内部
        if len(overlap_grain) == 1:
            continue
        # print(overlap_grain)
        # 提取对应晶粒
        for j in overlap_grain:
            # 单个晶粒区域进行计算和挑选，然后对符合条件的进行watershed操作，得到处理之后的局部切片
            water_info = water_grain(i, j, boundary_map, xcx_map, img, thresh_iou)
            # print("water_info", water_info)
            # 返回
            if water_info == None:  # 当前j grain不合适
                continue
            # 把处理之后的切片，贴回到boundary_map中去
            # 把分水岭之后的图像拼回去
            # 这里marker会把边界也给watershed，所以不能直接拼回去，要去掉边界。
            # 要注意判断 x和 y
            water_patch, x, y = water_info
            h, w = water_patch.shape
            step = 5
            x1 = x + step
            y1 = y + step
            h1 = h - step
            w1 = w - step
            # 把裁剪区域拼回去
            boundary_map[y1:y + h1, x1:x + w1] = water_patch[step:h1, step:w1]  # 这个拼接，好像拼小了？但应该不影响？

            # print("看看拼回去的值------------------------------------------------------------")
            # # 存一下看看是哪里出了问题：拼回去的label值不对应了
            save_color_boundary = color.label2rgb(boundary_map)  # 根据不同的标记显示不同的颜色
            # # print(np.unique(boundary_map))
            # io.imsave("E:\\1_Study\\Graduate_Project\\Coding\\gyy-water\\data\\step\\" + str(i) + "-" + str(j) + ".png", save_color_boundary)
    blank_ground = np.zeros(boundary_map.shape, dtype=np.uint8)
    label_b = segmentation.mark_boundaries(blank_ground, boundary_map, (255, 255, 255), background_label=0)
    # print("label_b.unique", np.unique(label_b))
    label_b[label_b > 0] == 1
    label_b = label_b.astype(np.uint8)
    label_b = skeletonize(label_b)

    # D_kernel = np.ones((5, 5), np.uint8)
    # label_b = cv2.dilate(label_b, D_kernel)
    label_b = cv2.cvtColor(label_b, cv2.COLOR_BGR2GRAY)
    water_img = dilate(label_b, 3)
    # 展示分割效果
    # skimageshow(water_img)

    return water_img


if __name__ == '__main__':
    today = datetime.date.today()
    thresh_iou = 0.9
    # 所要处理的数据文件夹，要求三个源数据文件夹的文件命名一致
    boundary_path = '/root/data/zhangxinyi/data/gyy-state2-train/post/OM/boundary'
    xcx_path = '/root/data/zhangxinyi/data/gyy-state2-train/post/OM/xcx'
    img_path =  '/root/data/zhangxinyi/data/gyy-state2-train/data/OM/test_img'
    save_path = '/root/data/zhangxinyi/data/gyy-state2-train/post/OM/water'+ '-' + str(thresh_iou)+"/"
    os.makedirs(save_path, mode=0o777, exist_ok=True) 

    total_time = []
    files = os.listdir(xcx_path)
    print(len(files))
    # print(files)
    # ok_files = os.listdir(save_path)
    # files = list(set(files) - set(ok_files))
    print(files)
    # 对单张图像进行处理
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
        # print(pic_boundary_path)
        
        t1 = time.time()
        
        water_img = water_post(pic_img_path, pic_boundary_path, pic_xcx_path, thresh_iou)
        
        t2 = (time.time()-t1)*1000
        print("单张图片推理时间：",t2)
        total_time.append(t2)

        # 后处理一下
        water_img = boundary_proc(water_img)
        water_img[water_img > 0] = 255
        # 保存图片
        cv2.imwrite(pic_save_path, water_img)

        plt.figure()
        plt.imshow(water_img)
        plt.show()
        
print("平均推理时间", sum(total_time)/len(total_time))
# %%
