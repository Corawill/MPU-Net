# coding:utf-8
import os

from skimage import io
from pathlib import Path
from IPython.display import clear_output
import matplotlib.pyplot as plt

import cv2, time
import random
import skimage
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as tr
from sklearn.model_selection import KFold
import pandas as pd

import metrics
from data_preprocess import *
from utils import *
from losser import *
from model.nets.unet import UNet
from model.nets.mpunet import MPUNet
from dataloader import SegDataset
from data_preprocess import *
import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# SegTrainer:Control the whole processing of training, validation and inference.  训练，验证，推理
class SegTrainer:
    """
        Segmentation class
    """
    # init：确定一些参数
# python语法： ->常常出现在python函数定义的函数名后面，为函数添加元数据,描述函数的返回类型，从而方便开发人员使用。下文为，本函数返回none
    def __init__(self)->None:
        # 自己新增的参数
        self.today = datetime.date.today()
        # model parameters
        self.model_name = "mpunet"  # unet resunet attunet, nestedunet，可用模型参数有四种
        self.model_class = MPUNet
        self.model = MPUNet()
        self.loss_name = "bcw"  # 'bcw', 'uw', 'dice'
        self.input_channels = 1 # 通道数为1，所以是灰度图像
        self.labels_class = 2

        # data parameters-
        self.dataset_name = "zxy-yw2021"
        #  
        self.data_experiment_path = None
        self.imgs_path = None
        self.labels_path = None
        self.labels_xcx_path = None
        self.weights_path = None
        self.files_name = None

        # 组合
        self.z_score_norm = tr.Compose([
            tr.ToTensor(),
        ])
        # 图像类型，suffix是添加后缀=============================
        self.imgs_suffix = "png"
        # 返回指定长度的字符串，不够的话前面填充0==============================================
        self.zfill_num = 3

        # training parameters
        self.seed_num = 2020  # 种子数目，设定了种子之后，训练集、验证集和测试集不会发生改变
        self.use_kfold = False # sklearn的交叉验证API用来计算交叉验证误差，让训练效果更好
        self.kf_num = 5 # n_splits 参数，用于把数据集分成5份
        # 这是啥？==========================================
        self.val_rate = 0.1 # 验证集的比率？=============================
        # 设定验证集度量标准，选择了loss，昨天马师兄讲了为什么选dice，因为用这个来进行结果比较的例子比较多
        self.val_metric_name = "loss"  # 'loss', 'dice', 'vi'
        # 检查是否为最小度量标准（loss或vi）metric（源代码拼写有错误）
        self.is_min_val_metirc = self.__check_min_val_metric()

        #训练批大小，决定梯度下降的速度和方向
        self.batch_size = 4  # at least 2，一般都是2的倍数，然后一般为2的幂

        self.crop_size = 512  # 128,256,512
        # 优化器衰减率
        self.weight_decay = 1e-3
        # 学习比率
        self.learning_rate = 1e-4  #后50轮为5e-5,1e-4
        # 练多少轮
        self.epochs = 60  # 10，先练一轮试试，60轮就收敛的可以了
        # 是否进行数据增强
        self.use_augment = [True, True, True]  # In training, rand_rotation, rand_vertical_flip, rand_horizontal_flip
        self.no_augment = [False, False, False]  # In Validate and test
    #===========================================================？？？
        # 优化器
        self.optimizer = None
        #
        self.scheduler = None
        #
        self.ignore_epoch_num = 2
        #
        self.best_model_path = None

        # loss parameters
        self.losser = Losser()

        # gpu parameters
        self.gpu_detector = GpuDetector()
        try:
            self.gpu_detector.set_gpu_device(0)
        except AssertionError as e:
            print(e)

        # inference parameters
        self.test_input_size = 512  # overlap-tile parameters
        self.test_overlap_size = 16  # overlap-tile parameters

        self.test_batch_size = 2  # batch_size in inference stage can improve the efficiency for overlap-tile strategy
        self.test_aug_type = 0  # Test Time augmentation, aug_type = 0: no tta, 1: 4 variants, 2: 8 variants
        self.test_use_post_process = False  # whether or not use post processing in test time
        self.test_metric_name = "vi"  # metirc for test data, "dice" or "vi"

# 选择测定方法
    def __check_min_val_metric(self) -> bool:
        if self.val_metric_name in ["loss", "vi"]:
            return True
        else:
            return False

# 数据预处理:
    def data_preprocess(self, data_bk_path: Path, data_experiment_path: Path) -> None:
        print("Data Preprocess ... ")
        # 根据参数不同，函数名重载，将数据从tif解压出来，处理成png和npy
        data_preprocess(self.dataset_name, data_bk_path, data_experiment_path, zfill_num=self.zfill_num,
                        suffix=self.imgs_suffix)  # you can change this

        print('Generating data in {}'.format(str(data_experiment_path)))

        self.z_score_norm = tr.Compose([
            tr.ToTensor()
        ])

        # Calculate weight map
        # Balanced Class Weight, bcw, Unet Weight, uw
#         weights_func = ["bcw", "uw"]
        # boundary
        weights_func = ["bcw"]
        prepare_weight_map(self.labels_path, data_experiment_path, weights_func, self.labels_class)
        print("Generated weight map in {}".format(str(data_experiment_path)))
        # xcx
        weights_func = ["uw"]
        prepare_weight_map(self.labels_xcx_path, data_experiment_path, weights_func, self.labels_class)
        print("Generated weight map in {}".format(str(data_experiment_path)))
        print("Data Preprocess has done")
        
# 开始训练
    def start_training(self, data_bk_path: Path, data_experiment_path: Path, experiment_path: Path,
                       use_preprocess: bool = False) -> None:
        
        self.data_experiment_path = data_experiment_path
        self.imgs_path = Path(data_experiment_path, "images")
        self.labels_path = Path(data_experiment_path, "labels")
        self.labels_xcx_path = Path(data_experiment_path, "labels_xcx")
        if self.loss_name in ["bcw", "uw"]:# 查看定义的loss
            self.weights_path = Path(data_experiment_path, self.loss_name)
        else:# 默认loss方法为bcw，这个地方代码好像没有用bcw
            self.weights_path = Path(data_experiment_path, "bcw")
        if use_preprocess:
            self.data_preprocess(data_bk_path, data_experiment_path)
        self.files_name = [str(item.name) for item in list(self.imgs_path.glob("*.{}".format(self.imgs_suffix)))]
        # 使用KFold交叉验证
        kf = KFold(n_splits=self.kf_num, shuffle=True, random_state=self.seed_num)

        test_metrics = []
        # 输入对于本次实验的描述
        print("please input the description of this experiment:")
        description = "test"   #input()
        experiment_name =  str(self.today) + '-' + description + '-' + self.dataset_name + '_' + self.model_name + '_train_' + self.loss_name + \
                          '_val_' + self.val_metric_name + '_test_' + self.test_metric_name + '_' + str(self.learning_rate) + '_' + str(self.epochs)
        if self.use_kfold:
            experiment_name += '_' + str(self.kf_num) + '_fold'

        global_parameters_path = Path(experiment_path, experiment_name)
        global_parameters_path.mkdir(parents=True, exist_ok=True)
        logger = Logger(True, Path(global_parameters_path, "log.txt"))
        # 记录日志
        logger.log_print("Experiment: {}".format(experiment_name))
        # logger.log_print("mean = {}, std ={} and image num = {}".format(self.imgs_mean, self.imgs_std, self.imgs_num))
        print("start training!!!")
        # K-fold Cross Validation
        for kf_idx, (train_files, test_files) in enumerate(kf.split(self.files_name)):
            # Dataset split，数据划分
            # we use k-fold cv to produce train set and test set, and then sample some data (10%)
            # from train set as val set (sampling without replacement)
#源代码的数据划分
            val_files = random.sample(list(train_files), int(len(train_files) * self.val_rate) + 1)
            train_files = list(set(train_files).difference(set(val_files)))
# # 控制test一致，写死数据集划分

            train_files.sort()
            val_files.sort()

            train_files = [self.files_name[item] for item in train_files]
            val_files = [self.files_name[item] for item in val_files]
            test_files = [self.files_name[item] for item in test_files]

            print("This is my train_files", len(train_files))
            print(train_files)

            print("This is my val_files", len(val_files))
            print(val_files)

            print("This is my testfile", len(test_files))
            print(test_files)
            
            if self.use_kfold:
                temp_parameters_path = Path(global_parameters_path, str(self.kf_num) + '_fold_' + str(kf_idx))
                temp_parameters_path.mkdir(parents=True, exist_ok=True)
                logger.log_print("Cross Validation: {}".format(str(self.kf_num) + '_fold_' + str(kf_idx)))
            else:
                temp_parameters_path = global_parameters_path
                if kf_idx >= 1:
                    break

            train_dataset = SegDataset(self.data_experiment_path, train_files, self.weights_path,
                                       use_augment=self.use_augment, crop_size=self.crop_size, depth=None,
                                       norm_transform=self.z_score_norm)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)# 一个打乱的数据集
            train_one_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)# 单个训练的数据集BatchSize=1
            val_dataset = SegDataset(self.data_experiment_path, val_files, self.weights_path,
                                     use_augment=self.no_augment, crop_size=self.crop_size, depth=None,
                                     norm_transform=self.z_score_norm)
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

            # 定义模型，优化器，优化器学习率调整方式
            self.model = self.model_class(num_channels=self.input_channels, num_classes=self.labels_class)
            
            # 计算模型的参数量大小
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.log_print(f"******{self.model_name}********, Total parameters: {total_params}")

            
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=6, gamma=0.9)
            if torch.cuda.is_available():#
                self.model = nn.DataParallel(self.model).cuda()

            train_metric_list = []
            val_metric_list = []

            val_baseline = 0
            if self.is_min_val_metirc:
                val_baseline = 10000

            val_best_epoch = 0

            # Training

#             # # 加载预训练模型(默认有预训练模型哈
#             train_folder = "2022-11-01-new-boundary-zxy-yw2021_unet_train_bcw_val_loss_test_vi_0.0001_50"
#             pre_model_path = "./experiment_gangyanyuan/" + train_folder + "/best_model_state.pth"
#             self.model.load_state_dict(torch.load(pre_model_path), False)
#             logger.log_print("加载预训练模型：pre_model_path: {} \n".format(pre_model_path))

            # 
            logger.log_print("不加载预训练模型：Train from scrath")

            st_total = time.time()
            logger.log_print("Training:")
            alpha = 2
            # 每一轮训练的内容
            for i in range(1, self.epochs + 1):
                print("Experiment name: " + experiment_name)
                st = time.time()
                self.__train(train_loader, alpha)
                if i%10 == 0:
                    alpha = alpha * 0.9
                    logger.log_print("alpha:{}".format(alpha))
                train_metric, metric_b, metric_x = self.__val(i, train_one_loader, temp_parameters_path, logger, check_first_img=False)
                self.writer.add_scalar('Boundary Loss/train', metric_b, i)
                self.writer.add_scalar('XCX Loss/train', metric_x, i)                
                
                val_metric, val_metric_b, val_metric_x = self.__val(i, val_loader, temp_parameters_path, logger, check_first_img=True)
                self.writer.add_scalar('Boundary Loss/val', val_metric_b, i)
                self.writer.add_scalar('XCX Loss/val', val_metric_x, i)
                
                train_metric_list.append(train_metric)
                val_metric_list.append(val_metric)

                logger.log_print(
                    "Epoch {}: train_{} {:.4f}; val_{} {:.4f} \n".format(i, self.val_metric_name, train_metric_list[-1],
                                                                         self.val_metric_name, val_metric_list[-1]))
                # 画出对应的图像
                plot(i, train_metric_list, val_metric_list, temp_parameters_path, self.is_min_val_metirc,
                     curve_name=self.val_metric_name, ignore_epoch=self.ignore_epoch_num)
                # 每轮比较参数，筛选最好的模型留下
                if (i > self.ignore_epoch_num) and ((self.is_min_val_metirc and val_metric < val_baseline) or (
                        self.is_min_val_metirc is False and val_metric > val_baseline)):
                    val_baseline = val_metric
                    val_best_epoch = i
                    torch.save(self.model.state_dict(), str(Path(temp_parameters_path, "best_model_state.pth")))
                ed = time.time()
                logger.log_print("Epoch Duration: {}'s".format(ed - st))
            # 记录训练结果
            ed_total = time.time()
            logger.log_print("Total duration is: {}'s".format(ed_total - st_total))
            logger.log_print("The best epoch is at: {} th epoch".format(val_best_epoch))
            logger.log_print("Train {} list is: {}".format(self.val_metric_name, train_metric_list))
            logger.log_print("Val   {} list is: {}".format(self.val_metric_name, val_metric_list))
            
            self.writer.close()
            
            self.best_model_path = Path(temp_parameters_path, "best_model_state.pth")

    def __train(self, dataloader, alpha):
        self.model.train()
        for sample in dataloader:  # 2D [B, C, H, W]
            # print(sample)
            if torch.cuda.is_available():
                img, label, xcx = sample['img'].cuda(), sample['label'].cuda(), sample['xcx'].cuda()
                if self.weights_path is not None:
                    weight = sample['weight'].cuda()
                    xcx_weight = sample['xcx_weight'].cuda()
            else:
                img, label, xcx = sample['img'], sample['label'], sample['xcx']
                if self.weights_path is not None:
                    weight = sample['weight']
                    xcx_weight = sample['xcx_weight'].cuda()
                    
            # print(self.model)
            output_b, output_x = self.model.forward(img)

            # 分别计算两个loss
            loss_b = self.losser.get_loss(output_b, label, weight, 'bcw', self.labels_class)
            loss_x = self.losser.get_loss(output_x, xcx, xcx_weight, 'uw', self.labels_class)
            # # 得到的结果相加，进行更新
            # print("loss_b:",loss_b)
            # print("loss_x:",loss_x)
            
            loss = alpha * loss_b + loss_x
            self.optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪
            nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()
        self.scheduler.step()
#todo
    def __val(self, epoch: int, dataloader, checkpoint_path: Path, logger, check_first_img=True) -> float:
        self.model.eval()
        val_metric = 0
        is_first = True
        metric_b = 0
        metric_x = 0
        # 反向传播时不用自动求导，节约显存和内存
        with torch.no_grad():
            for sample in dataloader:
                # if torch.cuda.is_available():
                if torch.cuda.is_available():
                    img, label, xcx = sample['img'].cuda(), sample['label'].cuda(), sample['xcx'].cuda()
                if self.weights_path is not None:
                    weight = sample['weight'].cuda()
                    xcx_weight = sample['xcx_weight'].cuda()
                else:
                    img, label, xcx = sample['img'], sample['label'], sample['xcx']
                    if self.weights_path is not None:
                        weight = sample['weight']
                        xcx_weight = sample['xcx_weight'].cuda()
                
                outputs = self.model.forward(img)

                # 返回两个值
                output_b, output_x = outputs
                
                temp_metric = 0

                if self.val_metric_name == 'loss':
                    loss_b = self.losser.get_loss(output_b, label, weight, self.loss_name, self.labels_class).item()
                    loss_x = self.losser.get_loss(output_x, xcx, xcx_weight, self.loss_name, self.labels_class).item()
                    temp_metric = loss_b + loss_x
                    # temp_metric += self.losser.get_loss(output_x, xcx, xcx_weight, self.loss_name, self.labels_class).item()
                    metric_b += loss_b
                    metric_x += loss_x
                else:
                    output = output_raw.max(1)[1].data
                    output = output.cpu().squeeze().numpy()
                    label = sample['label'].squeeze().numpy().astype(np.int64)
                    temp_metric = metrics.get_metric(self.val_metric_name, output, label)

                val_metric += temp_metric
                # visualization，可视化
                if check_first_img and is_first and epoch % 1 == 0:
                    check_result(epoch, sample['img'][0, :, :, :].numpy().squeeze(),
                                 sample['label'].numpy().squeeze(),
                                 sample['xcx'].numpy().squeeze(),
                                 output_b.max(1)[1].data.cpu().numpy().squeeze(),
                                 output_x.max(1)[1].data.cpu().numpy().squeeze(),
                                 sample['weight'][0, :, :, :].squeeze().numpy().transpose((1, 2, 0)),
                                 sample['xcx_weight'][0, :, :, :].squeeze().numpy().transpose((1, 2, 0)),
                                 checkpoint_path, logger)
                is_first = False
            val_metric /= len(dataloader.dataset)
            metric_b /= len(dataloader.dataset)
            metric_x /= len(dataloader.dataset)
            return val_metric, metric_b, metric_x
        
    
if __name__ == '__main__':
    seg_trainer = SegTrainer()
    # 返回当前工作目录
    cwd = os.getcwd()
    test_path = ""
    dataset_name = "zxy-yw2021"
    data_bk_path = Path(cwd, "data", dataset_name, "data_backup")
    data_experiment_path = Path(cwd, "data", dataset_name, "data_experiment")
    parameters_path = Path(cwd, "experiment")

    data_gangyanyuan_path = Path(cwd, "data", "FESEM")
       
    use_preprocess = False #第一轮为true，然后设定mean，num，std就好了
    seg_trainer.start_training(data_bk_path, data_gangyanyuan_path, parameters_path, use_preprocess)