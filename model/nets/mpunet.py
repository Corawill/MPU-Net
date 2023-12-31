# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import torch
import time

# *********************************** MPUNet *********************************************

class Encoder(nn.Module):
    def __init__(self, num_channels=1):
        super(Encoder, self).__init__()
        num_feat = [64, 128, 256, 512, 1024]

        self.down1 = nn.Sequential(Conv3x3(num_channels, num_feat[0]))

        self.down2 = nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                   Conv3x3(num_feat[0], num_feat[1]))

        self.down3 = nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                   Conv3x3(num_feat[1], num_feat[2]))

        self.down4 = nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                   Conv3x3(num_feat[2], num_feat[3]))

        self.bottom = nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                    Conv3x3(num_feat[3], num_feat[4]))

    def forward(self, inputs):
        down1_feat = self.down1(inputs)
        down2_feat = self.down2(down1_feat)
        down3_feat = self.down3(down2_feat)
        down4_feat = self.down4(down3_feat)
        bottom_feat = self.bottom(down4_feat)

        return bottom_feat


class Decoder(nn.Module):
    def __init__(self, num_classes=2):
        super(Decoder, self).__init__()
        num_feat = [64, 128, 256, 512, 1024]

        self.upconcat1 = UpConcat(num_feat[4], num_feat[3])
        self.upconv1 = Conv3x3(num_feat[4], num_feat[3])

        self.upconcat2 = UpConcat(num_feat[3], num_feat[2])
        self.upconv2 = Conv3x3(num_feat[3], num_feat[2])

        self.upconcat3 = UpConcat(num_feat[2], num_feat[1])
        self.upconv3 = Conv3x3(num_feat[2], num_feat[1])

        self.upconcat4 = UpConcat(num_feat[1], num_feat[0])
        self.upconv4 = Conv3x3(num_feat[1], num_feat[0])

        self.final = nn.Sequential(nn.Conv2d(num_feat[0], num_classes, kernel_size=1))

    def forward(self, bottom_feat, down1_feat, down2_feat, down3_feat, down4_feat):
        up1_feat = self.upconcat1(bottom_feat, down4_feat)
        up1_feat = self.upconv1(up1_feat)
        up2_feat = self.upconcat2(up1_feat, down3_feat)
        up2_feat = self.upconv2(up2_feat)
        up3_feat = self.upconcat3(up2_feat, down2_feat)
        up3_feat = self.upconv3(up3_feat)
        up4_feat = self.upconcat4(up3_feat, down1_feat)
        up4_feat = self.upconv4(up4_feat)

        outputs = self.final(up4_feat)

        return outputs


class MPUNet(nn.Module):
    def __init__(self, num_channels=1, num_classes=2):
        super(MPUNet, self).__init__()

        self.encoder = Encoder(num_channels=num_channels)
        self.decoder1 = Decoder(num_classes=num_classes)
        self.decoder2 = Decoder(num_classes=num_classes)

    def forward(self, inputs):
        # t1 = time.time()
        # bottom_feat = self.encoder(inputs) # 这里导致encoder多跑了一遍啊啊啊
        down1_feat = self.encoder.down1(inputs)
        down2_feat = self.encoder.down2(down1_feat)
        down3_feat = self.encoder.down3(down2_feat)
        down4_feat = self.encoder.down4(down3_feat)
        bottom_feat = self.encoder.bottom(down4_feat)
        # t2 = time.time()
        # print("encoder time:", t2-t1)
        outputs1 = self.decoder1(bottom_feat, down1_feat, down2_feat, down3_feat, down4_feat)
        # t3 = time.time()
        # print("decoder1 time:", t3-t2)
        outputs2 = self.decoder2(bottom_feat, down1_feat, down2_feat, down3_feat, down4_feat)
        # t4 = time.time()
        # print("decoder2 time:", t4-t3)

        return outputs1, outputs2



class Conv3x3(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(Conv3x3, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_feat, out_feat,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1),
                                   nn.BatchNorm2d(out_feat),
                                   nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv2d(out_feat, out_feat,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1),
                                   nn.BatchNorm2d(out_feat),
                                   nn.ReLU())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class UpConcat(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(UpConcat, self).__init__()

        self.deconv = nn.ConvTranspose2d(in_feat,
                                         out_feat,
                                         kernel_size=2,
                                         stride=2)

    def forward(self, inputs, down_outputs):
        outputs = self.deconv(inputs)
        out = torch.cat([down_outputs, outputs], 1)
        return out


if __name__ == "__main__":
    t = MPUNet()
    print(t)