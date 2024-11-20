import os
import json
import argparse
import tensorrt as trt

TRT_LOGGER = trt.Logger()

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)







if __name__ == '__main__':
    # Add plugins if needed
    # import ctypes
    # ctypes.CDLL("libmmdeploy_tensorrt_ops.so")
    parser = argparse.ArgumentParser(description='Writing qparams to onnx to convert tensorrt engine.')
    parser.add_argument('--onnx', type=str, default=None)
    parser.add_argument('--qparam_json', type=str, default=None)
    parser.add_argument('--engine', type=str, default=None)
    arg = parser.parse_args()

    build_engine(arg.onnx, arg.qparam_json, arg.engine)
    print("\033[1;32mgenerate %s\033[0m" % arg.engine)