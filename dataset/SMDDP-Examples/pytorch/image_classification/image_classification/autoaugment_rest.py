from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import random


class AutoaugmentImageNetPolicy(object):
    """
    Randomly choose one of the best 24 Sub-policies on ImageNet.
    Reference: https://arxiv.org/abs/1805.09501
    """




class SubPolicy(object):



class OperationFactory:



        self.operations = {
            "shearX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude * random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * random.choice([-1, 1])),
                fillcolor=fillcolor),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(magnitude),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "solarizeadd": lambda img, magnitude: solarize_add(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(magnitude),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(magnitude),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(magnitude),
            "autocontrast": lambda img, _: ImageOps.autocontrast(img),
            "equalize": lambda img, _: ImageOps.equalize(img),
            "invert": lambda img, _: ImageOps.invert(img)
        }

    def get_operation(self, method, magnitude_idx):
        magnitude = self.ranges[method][magnitude_idx]
        return lambda img: self.operations[method](img, magnitude)