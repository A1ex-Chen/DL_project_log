# reference: https://github.com/tjiiv-cprg/visdrone-det-toolkit-python


import os.path as osp
import os 
import numpy as np
import cv2
from viseval.eval_det import eval_det

import argparse
parser = argparse.ArgumentParser(description='Arguments')
parser.add_argument('--dataset-dir', required=True, type=str, help='output txt dir')
parser.add_argument('--res-dir', required=True, type=str, help='Grond Truth Info JSON')
args = parser.parse_args()





if __name__ == '__main__':
    main()