import os
import sys
from pathlib import Path
sys.path.append(os.path.abspath(Path(__file__).parent.parent))

import shutil
import cv2
import json
from visdrone import utils
from tqdm import tqdm

import argparse 


















if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Data Prepare Arguments')
    parser.add_argument('--visdrone-root', required=True, type=str, help='VisDrone dataset root')
    args = parser.parse_args()

    if not os.path.isdir(os.path.join(args.visdrone_root, 'coco_format')):
        os.mkdir(os.path.join(args.visdrone_root, 'coco_format'))
        os.mkdir(os.path.join(args.visdrone_root, 'coco_format/train_images'))
        os.mkdir(os.path.join(args.visdrone_root, 'coco_format/val_images'))
        os.mkdir(os.path.join(args.visdrone_root, 'coco_format/annotations'))
    

    '''
    Training
    '''
    train_img_root = os.path.join(args.visdrone_root, 'VisDrone2019-DET-train/images')
    train_label_root = os.path.join(args.visdrone_root, 'VisDrone2019-DET-train/annotations') 
    train_new_img_root = os.path.join(args.visdrone_root, 'coco_format/train_images')
    train_new_label_json = os.path.join(args.visdrone_root, 'coco_format/annotations/train_label.json') 
    make_new_train_set(train_img_root, train_label_root, train_new_img_root, train_new_label_json)

    '''
    Validation
    '''
    val_img_root = os.path.join(args.visdrone_root, 'VisDrone2019-DET-val/images') 
    val_label_root = os.path.join(args.visdrone_root, 'VisDrone2019-DET-val/annotations')   
    val_new_img_root = os.path.join(args.visdrone_root, 'coco_format/val_images')
    val_new_label_json = os.path.join(args.visdrone_root, 'coco_format/annotations/val_label.json') 
    make_new_test_set(val_img_root, val_label_root, val_new_img_root, val_new_label_json)

    '''
    Test set, not needed here. You can convert by yourself in the same way as validation set if you want to.  
    '''
    # img_root = '/path/to/test/images'
    # label_root = '/path/to/test/annotations'
    # new_img_root = '/path/to/test/images'
    # new_label_json = '/path/to/test/label.json'
    # make_new_test_set(img_root, label_root, new_img_root, new_label_json)

    

    
