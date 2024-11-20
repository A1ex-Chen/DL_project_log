import os.path

import onnx
import numpy as np
import struct
import sys
import copy







        # new_dequant_node = onnx.helper.make_node('DequantizeLinear',
        #                                         inputs=quant_node.input,
        #                                         outputs=prev_add_node.output,
        #                                         name=prev_add_node.name + "_DequantizeLinear")





if __name__ == '__main__':

    onnx_file = sys.argv[1]
    get_remove_qdq_onnx_and_cache(onnx_file)