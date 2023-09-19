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
from sklearn.model_selection import KFold, train_test_split
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

# SegTrainer:Control the whole processing of training, validation and inference.  
class SegTrainer:
    """
        Segmentation class
    """
    def __init__(self)->None:
        # model parameters
        self.today = datetime.date.today()
        self.model_name = "mpunet"  
        self.model_class = MPUNet
        self.model = MPUNet()
        self.loss_name = "bcw"  # 'bcw', 'uw'
        self.input_channels = 1 
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

        self.z_score_norm = tr.Compose([
            tr.ToTensor(),
        ])

        self.imgs_suffix = "png"

        self.zfill_num = 3

        # training parameters
        self.seed_num = 2020  
        self.kf_num = 5 
        self.val_rate = 0.1 
        self.val_metric_name = "loss" 
        self.is_min_val_metirc = self.__check_min_val_metric()

        self.batch_size = 4  # at least 2

        self.crop_size = 512  # 128,256,512

        self.weight_decay = 1e-3
 
        self.learning_rate = 1e-4 

        self.epochs = 60  

        self.use_augment = [True, True, True]  # In training, rand_rotation, rand_vertical_flip, rand_horizontal_flip
        self.no_augment = [False, False, False]  # In Validate and test

        self.optimizer = None
        
        self.scheduler = None
        
        self.ignore_epoch_num = 2
        
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


    def __check_min_val_metric(self) -> bool:
        if self.val_metric_name in ["loss", "vi"]:
            return True
        else:
            return False

    def data_preprocess(self, data_experiment_path: Path) -> None:
        print("Data Preprocess ... ")
        print('Generating data in {}'.format(str(data_experiment_path)))

        self.z_score_norm = tr.Compose([
            tr.ToTensor()
        ])

        # Calculate weight map
        # Balanced Class Weight, bcw, Unet Weight, uw
        # boundary
        weights_func = ["bcw"]
        prepare_weight_map(self.labels_path, data_experiment_path, weights_func, self.labels_class)
        print("Generated weight map in {}".format(str(data_experiment_path)))
        # xcx
        weights_func = ["uw"]
        prepare_weight_map(self.labels_xcx_path, data_experiment_path, weights_func, self.labels_class)
        print("Generated weight map in {}".format(str(data_experiment_path)))
        print("Data Preprocess has done")
        
    def start_training(self, data_experiment_path: Path, experiment_path: Path,
                       use_preprocess: bool = False) -> None:
        
        self.data_experiment_path = data_experiment_path
        self.imgs_path = Path(data_experiment_path, "images")
        print("imgs_path:------------------------------------",self.imgs_path)
        self.labels_path = Path(data_experiment_path, "labels")
        self.labels_xcx_path = Path(data_experiment_path, "labels_xcx")
        if self.loss_name in ["bcw", "uw"]:
            self.weights_path = Path(data_experiment_path, self.loss_name)
        else:
            self.weights_path = Path(data_experiment_path, "bcw")
        if use_preprocess:
            self.data_preprocess(data_experiment_path)
        # print(list(self.imgs_path.glob("*.*")))
        # print(list(self.imgs_path.glob("*.{}".format(self.imgs_suffix))))
        self.files_name = [str(item.name) for item in list(self.imgs_path.glob("*.{}".format(self.imgs_suffix)))]

        test_metrics = []
        print("please input the description of this experiment:")
        description = "test"   #input()
        experiment_name =  str(self.today) + '-' + description + '-' + self.dataset_name + '_' + self.model_name + '_train_' + self.loss_name + \
                          '_val_' + self.val_metric_name + '_test_' + self.test_metric_name + '_' + str(self.learning_rate) + '_' + str(self.epochs)

        global_parameters_path = Path(experiment_path, experiment_name)
        global_parameters_path.mkdir(parents=True, exist_ok=True)
        logger = Logger(True, Path(global_parameters_path, "log.txt"))
        logger.log_print("Experiment: {}".format(experiment_name))
        print("start training!!!")

        n_samples = len(self.files_name)
        # print("len:files:",n_samples)
        # Dataset split
        train_files, test_files = train_test_split(self.files_name, test_size=0.2, train_size= 0.8, random_state=self.seed_num)
        # print(train_files)
        val_files = random.sample(list(train_files), int(len(train_files) * self.val_rate) + 1)
        train_files = list(set(train_files).difference(set(val_files)))

        train_files.sort()
        val_files.sort()

        # train_files = [self.files_name[item] for item in train_files]
        # val_files = [self.files_name[item] for item in val_files]
        # test_files = [self.files_name[item] for item in test_files]

        print("This is my train_files", len(train_files))
        print(train_files)

        print("This is my val_files", len(val_files))
        print(val_files)

        print("This is my testfile", len(test_files))
        print(test_files)
        
        temp_parameters_path = global_parameters_path

        train_dataset = SegDataset(self.data_experiment_path, train_files, self.weights_path,
                                    use_augment=self.use_augment, crop_size=self.crop_size, depth=None,
                                    norm_transform=self.z_score_norm)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        train_one_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        val_dataset = SegDataset(self.data_experiment_path, val_files, self.weights_path,
                                    use_augment=self.no_augment, crop_size=self.crop_size, depth=None,
                                    norm_transform=self.z_score_norm)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        self.model = self.model_class(num_channels=self.input_channels, num_classes=self.labels_class)
        
        # Calculate the parameter size of the model
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
#             # Load pretrained model
#             train_folder = " "
#             pre_model_path = "./experiment_gangyanyuan/" + train_folder + "/best_model_state.pth"
#             self.model.load_state_dict(torch.load(pre_model_path), False)
#             logger.log_print("Load pretrained model：pre_model_path: {} \n".format(pre_model_path))

        logger.log_print("Train from scrath")

        st_total = time.time()
        logger.log_print("Training:")
        alpha = 2
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

            plot(i, train_metric_list, val_metric_list, temp_parameters_path, self.is_min_val_metirc,
                    curve_name=self.val_metric_name, ignore_epoch=self.ignore_epoch_num)

            if (i > self.ignore_epoch_num) and ((self.is_min_val_metirc and val_metric < val_baseline) or (
                    self.is_min_val_metirc is False and val_metric > val_baseline)):
                val_baseline = val_metric
                val_best_epoch = i
                torch.save(self.model.state_dict(), str(Path(temp_parameters_path, "best_model_state.pth")))
            ed = time.time()
            logger.log_print("Epoch Duration: {}'s".format(ed - st))

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

            loss_b = self.losser.get_loss(output_b, label, weight, 'bcw', self.labels_class)
            loss_x = self.losser.get_loss(output_x, xcx, xcx_weight, 'uw', self.labels_class)
            
            loss = alpha * loss_b + loss_x
            self.optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()
        self.scheduler.step()

    def __val(self, epoch: int, dataloader, checkpoint_path: Path, logger, check_first_img=True) -> float:
        self.model.eval()
        val_metric = 0
        is_first = True
        metric_b = 0
        metric_x = 0
        with torch.no_grad():
            for sample in dataloader:
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

                output_b, output_x = outputs
                
                temp_metric = 0

                if self.val_metric_name == 'loss':
                    loss_b = self.losser.get_loss(output_b, label, weight, self.loss_name, self.labels_class).item()
                    loss_x = self.losser.get_loss(output_x, xcx, xcx_weight, self.loss_name, self.labels_class).item()
                    temp_metric = loss_b + loss_x
                    # temp_metric += self.losser.get_loss(output_x, xcx, xcx_weight, self.loss_name, self.labels_class).item()
                    metric_b += loss_b
                    metric_x += loss_x

                val_metric += temp_metric
                # visualization
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

    cwd = os.getcwd()
    data_path = Path(cwd, "data", "FESEM")
    parameters_path = Path(cwd, "experiment")
       
    use_preprocess = False # Set to True in the first round to generate the weight map
    seg_trainer.start_training(data_path, parameters_path, use_preprocess)