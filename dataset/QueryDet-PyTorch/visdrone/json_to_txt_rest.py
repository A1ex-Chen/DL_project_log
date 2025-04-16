import os
import tqdm
import json


import argparse


class Json2Txt(object):

    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('--out', required=True, type=str, help='output txt dir')
    parser.add_argument('--gt-json', required=False, type=str, default='visdrone_data/annotations/val_label', help='Grond Truth Info JSON')
    parser.add_argument('--det-json', required=True, type=str, help='COCO style result JSON')
    args = parser.parse_args()

    gt_json = args.gt_json
    det_json = args.det_json
    outdir = args.out
    
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
        
    print('Json to txt:', outdir)
    tool = Json2Txt(gt_json, det_json, outdir)
    tool.to_txt()
    









































