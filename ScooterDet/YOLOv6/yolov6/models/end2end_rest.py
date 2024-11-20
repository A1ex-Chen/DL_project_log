import torch
import torch.nn as nn
import random


class ORT_NMS(torch.autograd.Function):
    '''ONNX-Runtime NMS operation'''
    @staticmethod

    @staticmethod


class TRT8_NMS(torch.autograd.Function):
    '''TensorRT NMS operation'''
    @staticmethod

    @staticmethod

class TRT7_NMS(torch.autograd.Function):
    '''TensorRT NMS operation'''
    @staticmethod
    @staticmethod


class ONNX_ORT(nn.Module):
    '''onnx module with ONNX-Runtime NMS operation.'''


class ONNX_TRT7(nn.Module):
    '''onnx module with TensorRT NMS operation.'''


class ONNX_TRT8(nn.Module):
    '''onnx module with TensorRT NMS operation.'''



class End2End(nn.Module):
    '''export onnx or tensorrt model with NMS operation.'''
