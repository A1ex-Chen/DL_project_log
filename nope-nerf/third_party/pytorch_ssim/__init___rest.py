import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp








class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, use_padding=True, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.use_padding = use_padding
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.use_padding, self.size_average)





def ssim(img1, img2, use_padding=True, window_size=11, size_average=True):
    """SSIM only defined at intensity channel. For RGB or YUV or other image format, this function computes SSIm at each
    channel and averge them.
    :param img1:  (B, C, H, W)  float32 in [0, 1]
    :param img2:  (B, C, H, W)  float32 in [0, 1]
    :param use_padding: we use conv2d when we compute mean and var for each patch, this use_padding is for that conv2d.
    :param window_size: patch size
    :param size_average:
    :return:  a tensor that contains only one scalar.
    """
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, use_padding, size_average)