from importlib.resources import path
import matplotlib as mpl
# we cannot use remote server's GUI, so set this  
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm as CM
from PIL import Image
import h5py
import numpy as np
import cv2

import skimage.io
from glob import glob
import os

import math
import torch
import torchvision.transforms as transforms
from src.args import ArgumentParserRGBDSegmentation
from src.build_model import build_model #?build_model作为我的模型，与ArgumentParserRGBDSegmentation默认参数配合，得到模型
import argparse

from keras.models import *
from keras.layers import *






if __name__ == "__main__":
    
   
    get_attention()

# ***************************************************************************************************************
# import matplotlib as mpl
# # we cannot use remote server's GUI, so set this  
# mpl.use('Agg')
# import matplotlib.pyplot as plt
# from matplotlib import cm as CM
# from PIL import Image
# import h5py
# import numpy as np
# import cv2

# root_path = '/home/cyxiong/SGACNet1/samples/rgb/'
# base_path='/home/cyxiong/SGACNet1/samples/feature/nyuv2/'

# f=open("/home/cyxiong/SGACNet1/samples/feature/nyuv2/nyuv2.txt","r") #line  feature_map f
# p=open("/home/cyxiong/SGACNet1/samples/rgb/rgb.txt","r")                #line1  rgb_map p
# while True:
#             line=f.readline() #and p.readline() #包括换行符  line=f.readline()
#             line =line [:-1]     #去掉换行符
#             line1=p.readline() #and p.readline() #包括换行符  line=f.readline()
#             line1 =line1 [:-1]     #去掉换行符
#             if line and line1:
#                 print (line)
#                 print (line1)
#                 rgb_path = root_path+line1+'.png'
#                 feature_path=base_path+line
#                 img_rgb=cv2.imread(rgb_path)
#                 img_feature=cv2.imread(feature_path)
#                 # adaptive gaussian filter
                