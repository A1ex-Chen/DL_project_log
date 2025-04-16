# import matplotlib.pyplot as plt
# import numpy as np
 
# #epoch,acc,loss,val_acc,val_loss
# x_axis_data = [1,2,3,4,5,6,7]
# y_axis_data1 = [68.72,69.17,69.26,69.63,69.35,70.3,66.8]
# y_axis_data2 = [71,73,52,66,74,82,71]
# y_axis_data3 = [82,83,82,76,84,92,81]

        
# #画图 
# plt.plot(x_axis_data, y_axis_data1, 'b*--', alpha=0.5, linewidth=1, label='ResNet18')#'#'bo-'表示蓝色实线，数据点实心原点标注
# ## plot中参数的含义分别是横轴值，纵轴值，线的形状（'s'方块,'o'实心圆点，'*'五角星   ...，颜色，透明度,线的宽度和标签 
# plt.plot(x_axis_data, y_axis_data2, 'rs--', alpha=0.5, linewidth=1, label='acc')
# plt.plot(x_axis_data, y_axis_data3, 'go--', alpha=0.5, linewidth=1, label='acc')
# # #* plot中参数的含义分别是横轴值，纵轴值，线的形状，颜色，透明度,线的宽度和标签
# # *plt.plot(x_axis_data, y_axis_data, 'ro-', color='#4169E1', alpha=0.8, linewidth=1, label='ResNet18')
 
# plt.legend()  #显示上面的label
# plt.xlabel('FPS ')   #*FPS (NVIDIA Jetson AGX Xavier, TensorRT 7.1, Float16) NYUv2 #*x轴数字
# plt.ylabel('mIoU(%)')  #accuracy


# #plt.ylim(-1,1)#仅设置y轴坐标范围
# plt.show()




# # 显示标签，如果不加这句，即使在plot中加了label='一些数字'的参数，最终还是不会显示标签
# plt.legend(loc="upper right")
# import torchvision
# import matplotlib.pyplot as plt
# import torch
# import torch.nn as nn
# from torch.nn import functional as F
# from torchvision import transforms
# import numpy as np
# from PIL import Image
# from collections import OrderedDict
# import cv2

# import argparse
# from src.build_model import build_model
# from src.prepare_data import prepare_data
# from src.args import ArgumentParserRGBDSegmentation






# if __name__ == '__main__':
# # *arguments
#     parser = ArgumentParserRGBDSegmentation(
#     description='Efficient RGBD Indoor Sematic Segmentation (Inference)',
#     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.set_common_args()
#     # parser.add_argument('--ckpt_path', type=str,
#     #             required=True,
#     #             help='Path to the checkpoint of the trained model.')
#     # parser.add_argument('--depth_scale', type=float,
#     #             default=1.0,
#     #             help='Additional depth scaling factor to apply.')

#     args = parser.parse_args()

#     #* dataset
#     # args.pretrained_on_imagenet = False  # we are loading other weights anyway
#     # dataset, preprocessor = prepare_data(args, with_input_orig=True)
#     # n_classes = dataset.n_classes_without_void

#     #* model and checkpoint loading
#     # model = torch.load('./save/model.pkl').to(torch.device('cpu'))
#     # print(model)
#     model = torch.load('/home/cyxiong/SGACNet/results/nyuv2/49.39-ORII-checkpoints_03_06_2022-20_42_10-189473/ckpt_epoch_488.pth').to(torch.device('cpu'))
#     # print(model)  #*AttributeError: 'dict' object has no attribute 'to'

#     # model, device = build_model(args, n_classes=n_classes)
#     checkpoint = torch.load(args.ckpt_path,
#                     map_location=lambda storage, loc: storage)

#     # state_dict  = checkpoint['state_dict']
#     # model.load_state_dict(state_dict,False)
#     model.load_state_dict(checkpoint['state_dict'])
#     print('Loaded checkpoint from {}'.format(args.ckpt_path))
    
#     model.to('cpu')
#     print(model)
    

# ############################################################*.pth包括rgb和depth，图片输入+融合，不匹配Missing key(s) in state_dict:########################################################

# from xml.parsers.expat import model
# import cv2
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
 
# import torch
# import torch.autograd as autograd
# import torchvision.transforms as transforms
# from src.build_model import build_model
# # from src.models.model import SGACNet
# from src.args import ArgumentParserRGBDSegmentation
# from src.prepare_data import prepare_data
# import argparse

# # 训练过的模型路径
# resume_path = r"/home/cyxiong/SGACNet/results/nyuv2/49.13-re-r34-checkpoints_31_07_2022-14_43_15-846351/ckpt_epoch_305.pth"

# # 输入图像路径
# single_img_path = r'/home/cyxiong/SGACNet/samples/Figure_1.jpg'

# # 绘制的热力图存储路径
# save_path = r'/home/cyxiong/SGACNet/samples/Fig1.jpg'

# # 网络层的层名列表, 需要根据实际使用网络进行修改
# layers_names = ['activation','encoder_rgb','encoder_depth','conv1','bn1','act', 'maxpool','encoder_depth','se_depth','se_layer0','conv1','se_layer1'
#                 ,'se_layer2', 'se_layer3','se_layer4','skip_layer1','skip_layer2','skip_layer3','context_module','relu', 'layer1', 
#                 'layer2', 'layer3', 'layer4', 'avgpool','decoder','side_output','conv_out','upsample','upsample1','upsample2']
# # 指定层名
# out_layer_name = "layer4"


# features_grad = 0
 
 
# # 为了读取模型中间参数变量的梯度而定义的辅助函数
# def extract(g):
#     global features_grad

#     features_grad = g
    
# def draw_CAM(model, img_path, save_path, transform=None, visual_heatmap=False, out_layer=None):
#     """
#     绘制 Class Activation Map
#     :param model: 加载好权重的Pytorch model
#     :param img_path: 测试图片路径
#     :param save_path: CAM结果保存路径
#     :param transform: 输入图像预处理方法
#     :param visual_heatmap: 是否可视化原始heatmap（调用matplotlib）
#     :return:
#     """
#     # 读取图像并预处理
#     global layer2
#     img = Image.open(img_path).convert('RGB')
#     if transform:
#         img = transform(img).cuda()
#     img = img.unsqueeze(0)  # (1, 3, 448, 448)
 
#     # model转为eval模式
#     model.eval()
 
#     # 获取模型层的字典
#     layers_dict = {layers_names[i]: None for i in range(len(layers_names))}
#     for i, (name, module) in enumerate(model.features._modules.items()):
#         layers_dict[layers_names[i]] = module
 
#     # 遍历模型的每一层, 获得指定层的输出特征图
#     # features: 指定层输出的特征图, features_flatten: 为继续完成前端传播而设置的变量
#     features = img
#     start_flatten = False
#     features_flatten = None
#     for name, layer in layers_dict.items():
#         if name != out_layer and start_flatten is False:    # 指定层之前
#             features = layer(features)
#         elif name == out_layer and start_flatten is False:  # 指定层
#             features = layer(features)
#             start_flatten = True
#         else:   # 指定层之后
#             if features_flatten is None:
#                 features_flatten = layer(features)
#             else:
#                 features_flatten = layer(features_flatten)
 
#     features_flatten = torch.flatten(features_flatten, 1)
#     output = model.classifier(features_flatten)
 
#     # 预测得分最高的那一类对应的输出score
#     pred = torch.argmax(output, 1).item()
#     pred_class = output[:, pred]
 
#     # 求中间变量features的梯度
#     # 方法1
#     # features.register_hook(extract)
#     # pred_class.backward()
#     # 方法2
#     features_grad = autograd.grad(pred_class, features, allow_unused=True)[0]
 
#     grads = features_grad  # 获取梯度
#     pooled_grads = torch.nn.AdaptiveAvgPool2d(grads, (1, 1))
#     # 此处batch size默认为1，所以去掉了第0维（batch size维）
#     pooled_grads = pooled_grads[0]
#     features = features[0]
#     print("pooled_grads:", pooled_grads.shape)
#     print("features:", features.shape)
#     # features.shape[0]是指定层feature的通道数
#     for i in range(features.shape[0]):
#         features[i, ...] *= pooled_grads[i, ...]
 
#     # 计算heatmap
#     heatmap = features.detach().cpu().numpy()
#     heatmap = np.mean(heatmap, axis=0)
#     heatmap = np.maximum(heatmap, 0)
#     heatmap /= np.max(heatmap)
 
#     # 可视化原始热力图
#     if visual_heatmap:
#         plt.matshow(heatmap)
#         plt.show()
 
#     img = cv2.imread(img_path)  # 用cv2加载原始图像
#     heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
#     heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
#     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
#     superimposed_img = heatmap * 0.7 + img  # 这里的0.4是热力图强度因子
#     cv2.imwrite(save_path, superimposed_img)  # 将图像保存到硬盘
 
 
# if __name__ == '__main__':
#        # arguments
#     parser = ArgumentParserRGBDSegmentation(
#         description='Efficient RGBD Indoor Sematic Segmentation (Evaluation)',
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.set_common_args()
#     parser.add_argument('--ckpt_path', type=str,
#                         required=True,
#                         help='Path to the checkpoint of the trained model.')
#     args = parser.parse_args()
    
#     transform = transforms.Compose([
#         transforms.Resize(448),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#     ])
#     # 构建模型并加载预训练参数
#     # seresnet34 = SGACNet(num_classes=2).cuda()
#     # checkpoint = torch.load(resume_path)
#     # seresnet34.load_state_dict(checkpoint['state_dict'])
    
#     # seresnet34 = SGACNet(num_classes=2).cuda() 
#     # checkpoint = torch.load(resume_path,
#     #                         map_location=lambda storage, loc: storage)
#     # seresnet34.load_state_dict(checkpoint['state_dict'])
    
    
    
#     # dataset
#     args.pretrained_on_imagenet = False  # we are loading other weights anyway
#     _, data_loader, *add_data_loader = prepare_data(args, with_input_orig=True)
#     if args.valid_full_res:
#         # cityscapes only -> use dataloader that returns full resolution images
#         data_loader = add_data_loader[0]

#     n_classes = data_loader.dataset.n_classes_without_void
#     # model and checkpoint loading
#     model, device = build_model(args, n_classes=n_classes)
#     checkpoint = torch.load(args.ckpt_path,
#                             map_location=lambda storage, loc: storage)
    
#     model.load_state_dict(checkpoint['state_dict'])
#     print('Loaded checkpoint from {}'.format(args.ckpt_path))
    
#     draw_CAM(model, single_img_path, save_path, transform=transform, visual_heatmap=True, out_layer=out_layer_name)
    
#     # draw_CAM(seresnet34, single_img_path, save_path, transform=transform, visual_heatmap=True, out_layer=out_layer_name)

# ############################################################*##############################################################################################################################
import os
import torch
import torchvision.transforms as transforms
import skimage.data
import skimage.io
import skimage.transform
import numpy as np
import matplotlib.pyplot as plt
# *from completion_segmentation_model import DepthCompletionFrontNet
#* from completion_segmentation_model_v3_eca_attention import DepthCompletionFrontNet
import math

from src.args import ArgumentParserRGBDSegmentation
from src.build_model import build_model #?build_model作为我的模型，与ArgumentParserRGBDSegmentation默认参数配合，得到模型
import argparse
from src.prepare_data import prepare_data

import imageio

import cv2
import torch.nn.functional as F
# from PIL import Image
#? *实现单通道图片进行读取并将其转化为Tensor  定义数据预处理方式(将输入的类似numpy中arrary形式的数据转化为pytorch中的张量（tensor）)
transform = transforms.ToTensor()

#? 定义是否使用GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  #*在后面model and checkpoint loading部分定义

# img = Image.open('E:/Dataset/kaist_pixel_level/Test/Thermal Images/set06_V000_I00019.jpg')
# img = np.array(img, dtype=np.uint8)
# img = transform(img)
# print(img.shape) # torch.Size([3, 512, 640])
# img = img.unsqueeze(0)
# print(img.shape) # torch.Size([1, 3, 512, 640])






    # f.close()
    # for line in open("/home/cyxiong/SGACNet/datasets/nyuv2/train.txt","r"): #设置文件对象并读取每一行文件
        
    #* pic_dir = '0058.png'
    
    # depth_path = root_path+'depth_raw/'+line+'.png'
    # rgb_path = root_path+'rgb/'+line+'.png'
    # picture_dir= root_path+'rgb/'+f+'.png'
    
    # *depth_path = root_path+'depth_raw/'+pic_dir
    # *rgb_path = root_path+'rgb/'+pic_dir
    # picture_dir= root_path+'rgb/'+pic_dir
    #     # preprocess sample
    # sample = preprocessor({'image': img_rgb, 'depth': img_depth})
    #  # add batch axis and copy to device
    # image = sample['image'][None].to(device)
    # depth = sample['depth'][None].to(device)

#* 训练
if __name__ == "__main__":
    
    
    get_feature()