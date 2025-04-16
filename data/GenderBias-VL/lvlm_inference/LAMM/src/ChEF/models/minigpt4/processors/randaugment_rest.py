"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

# import cv2
import numpy as np

import torch


## aug functions














### level to args















    channels = [tune_channel(ch) for ch in cv2.split(img)]
    out = cv2.merge(channels)
    return out


# def equalize_func(img):
    """
    same output as PIL.ImageOps.equalize
    PIL's implementation is different from cv2.equalize
    """
    n_bins = 256


    channels = [tune_channel(ch) for ch in cv2.split(img)]
    out = cv2.merge(channels)
    return out


# def rotate_func(img, degree, fill=(0, 0, 0)):
    """
    like PIL, rotate by degree, not radians
    """
    H, W = img.shape[0], img.shape[1]
    center = W / 2, H / 2
    M = cv2.getRotationMatrix2D(center, degree, 1)
    out = cv2.warpAffine(img, M, (W, H), borderValue=fill)
    return out


def solarize_func(img, thresh=128):
    """
    same output as PIL.ImageOps.posterize
    """
    table = np.array([el if el < thresh else 255 - el for el in range(256)])
    table = table.clip(0, 255).astype(np.uint8)
    out = table[img]
    return out


def color_func(img, factor):
    """
    same output as PIL.ImageEnhance.Color
    """
    ## implementation according to PIL definition, quite slow
    #  degenerate = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]
    #  out = blend(degenerate, img, factor)
    #  M = (
    #      np.eye(3) * factor
    #      + np.float32([0.114, 0.587, 0.299]).reshape(3, 1) * (1. - factor)
    #  )[np.newaxis, np.newaxis, :]
    M = np.float32(
        [[0.886, -0.114, -0.114], [-0.587, 0.413, -0.587], [-0.299, -0.299, 0.701]]
    ) * factor + np.float32([[0.114], [0.587], [0.299]])
    out = np.matmul(img, M).clip(0, 255).astype(np.uint8)
    return out


def contrast_func(img, factor):
    """
    same output as PIL.ImageEnhance.Contrast
    """
    mean = np.sum(np.mean(img, axis=(0, 1)) * np.array([0.114, 0.587, 0.299]))
    table = (
        np.array([(el - mean) * factor + mean for el in range(256)])
        .clip(0, 255)
        .astype(np.uint8)
    )
    out = table[img]
    return out


def brightness_func(img, factor):
    """
    same output as PIL.ImageEnhance.Contrast
    """
    table = (np.arange(256, dtype=np.float32) * factor).clip(0, 255).astype(np.uint8)
    out = table[img]
    return out


# def sharpness_func(img, factor):
    """
    The differences the this result and PIL are all on the 4 boundaries, the center
    areas are same
    """
    kernel = np.ones((3, 3), dtype=np.float32)
    kernel[1][1] = 5
    kernel /= 13
    degenerate = cv2.filter2D(img, -1, kernel)
    if factor == 0.0:
        out = degenerate
    elif factor == 1.0:
        out = img
    else:
        out = img.astype(np.float32)
        degenerate = degenerate.astype(np.float32)[1:-1, 1:-1, :]
        out[1:-1, 1:-1, :] = degenerate + factor * (out[1:-1, 1:-1, :] - degenerate)
        out = out.astype(np.uint8)
    return out


# def shear_x_func(img, factor, fill=(0, 0, 0)):
    H, W = img.shape[0], img.shape[1]
    M = np.float32([[1, factor, 0], [0, 1, 0]])
    out = cv2.warpAffine(
        img, M, (W, H), borderValue=fill, flags=cv2.INTER_LINEAR
    ).astype(np.uint8)
    return out


# def translate_x_func(img, offset, fill=(0, 0, 0)):
    """
    same output as PIL.Image.transform
    """
    H, W = img.shape[0], img.shape[1]
    M = np.float32([[1, 0, -offset], [0, 1, 0]])
    out = cv2.warpAffine(
        img, M, (W, H), borderValue=fill, flags=cv2.INTER_LINEAR
    ).astype(np.uint8)
    return out


# def translate_y_func(img, offset, fill=(0, 0, 0)):
    """
    same output as PIL.Image.transform
    """
    H, W = img.shape[0], img.shape[1]
    M = np.float32([[1, 0, 0], [0, 1, -offset]])
    out = cv2.warpAffine(
        img, M, (W, H), borderValue=fill, flags=cv2.INTER_LINEAR
    ).astype(np.uint8)
    return out


def posterize_func(img, bits):
    """
    same output as PIL.ImageOps.posterize
    """
    out = np.bitwise_and(img, np.uint8(255 << (8 - bits)))
    return out


# def shear_y_func(img, factor, fill=(0, 0, 0)):
    H, W = img.shape[0], img.shape[1]
    M = np.float32([[1, 0, 0], [factor, 1, 0]])
    out = cv2.warpAffine(
        img, M, (W, H), borderValue=fill, flags=cv2.INTER_LINEAR
    ).astype(np.uint8)
    return out


def cutout_func(img, pad_size, replace=(0, 0, 0)):
    replace = np.array(replace, dtype=np.uint8)
    H, W = img.shape[0], img.shape[1]
    rh, rw = np.random.random(2)
    pad_size = pad_size // 2
    ch, cw = int(rh * H), int(rw * W)
    x1, x2 = max(ch - pad_size, 0), min(ch + pad_size, H)
    y1, y2 = max(cw - pad_size, 0), min(cw + pad_size, W)
    out = img.copy()
    out[x1:x2, y1:y2, :] = replace
    return out


### level to args
def enhance_level_to_args(MAX_LEVEL):

    return level_to_args


def shear_level_to_args(MAX_LEVEL, replace_value):

    return level_to_args


def translate_level_to_args(translate_const, MAX_LEVEL, replace_value):

    return level_to_args


def cutout_level_to_args(cutout_const, MAX_LEVEL, replace_value):

    return level_to_args


def solarize_level_to_args(MAX_LEVEL):

    return level_to_args


def none_level_to_args(level):
    return ()


def posterize_level_to_args(MAX_LEVEL):

    return level_to_args


def rotate_level_to_args(MAX_LEVEL, replace_value):

    return level_to_args


func_dict = {
    "Identity": identity_func,
    # "AutoContrast": autocontrast_func,
    # "Equalize": equalize_func,
    # "Rotate": rotate_func,
    "Solarize": solarize_func,
    "Color": color_func,
    "Contrast": contrast_func,
    "Brightness": brightness_func,
    # "Sharpness": sharpness_func,
    # "ShearX": shear_x_func,
    # "TranslateX": translate_x_func,
    # "TranslateY": translate_y_func,
    "Posterize": posterize_func,
    # "ShearY": shear_y_func,
}

translate_const = 10
MAX_LEVEL = 10
replace_value = (128, 128, 128)
arg_dict = {
    "Identity": none_level_to_args,
    "AutoContrast": none_level_to_args,
    "Equalize": none_level_to_args,
    "Rotate": rotate_level_to_args(MAX_LEVEL, replace_value),
    "Solarize": solarize_level_to_args(MAX_LEVEL),
    "Color": enhance_level_to_args(MAX_LEVEL),
    "Contrast": enhance_level_to_args(MAX_LEVEL),
    "Brightness": enhance_level_to_args(MAX_LEVEL),
    "Sharpness": enhance_level_to_args(MAX_LEVEL),
    "ShearX": shear_level_to_args(MAX_LEVEL, replace_value),
    "TranslateX": translate_level_to_args(translate_const, MAX_LEVEL, replace_value),
    "TranslateY": translate_level_to_args(translate_const, MAX_LEVEL, replace_value),
    "Posterize": posterize_level_to_args(MAX_LEVEL),
    "ShearY": shear_level_to_args(MAX_LEVEL, replace_value),
}


class RandomAugment(object):




class VideoRandomAugment(object):





if __name__ == "__main__":
    a = RandomAugment()
    img = np.random.randn(32, 32, 3)
    a(img)