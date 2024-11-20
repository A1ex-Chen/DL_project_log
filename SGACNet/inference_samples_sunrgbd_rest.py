# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import argparse
from glob import glob
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from src.args import ArgumentParserRGBDSegmentation
from src.build_model import build_model
from src.prepare_data import prepare_data

import skimage.data
import skimage.io
import skimage.transform
import torchvision.transforms as transforms

# def _load_img(fp):
#     img = cv2.imread(fp, cv2.IMREAD_UNCHANGED) #*cv2.IMREAD_UNCHANGED：顾名思义，读入完整图片，包括alpha通道，可用-1作为实参替代
#     if img.ndim == 3:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     return img

#? *实现单通道图片进行读取并将其转化为Tensor  定义数据预处理方式(将输入的类似numpy中arrary形式的数据转化为pytorch中的张量（tensor）)
transform = transforms.ToTensor()



 # arguments
                # f.close()   



if __name__ == '__main__':
   
    get_results()
    
    
   
    
    
    
    
    
    

    # #######*#############################################################
    # rgb_filepaths = os.path.join(os.path.dirname(os.path.abspath(__file__)),
    #                         'depth_raw')
    # depth_filepaths = os.path.join(os.path.dirname(os.path.abspath(__file__)),
    #                         'labels_40_colored')
    
    
    # ###############*##########################################################################################
    # basepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
    #                         'datasets')
    # basepath =os.path.join(basepath,os.path.join('nyuv2','train'))
    # rgb_filepath = os.path.join(basepath,'depth_raw')
    # rgb_filepaths = sorted(glob(os.path.join(rgb_filepath, '*0032.*')))
    # depth_filepath= os.path.join(basepath,'labels_40_colored')
    # depth_filepaths = sorted(glob(os.path.join(depth_filepath, '*0032.*')))
    # # total_num = len(rgb_filepaths)  #*得到文件夹中图像的个数
# ###############*#####################################################################################################################

    