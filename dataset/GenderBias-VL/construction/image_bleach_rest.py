import argparse
import csv
from functools import partial
from io import TextIOWrapper
import os
import clip
from matplotlib import pyplot as plt
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
import random
from torch.utils.data import DataLoader

# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt

src_dir = "./xl_generate_base" ### base images directory
exp_dir = "./image_bleach" ### output directory



class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, transform, root_path=None,
                 occupation_path:str=None):
        self.root_path = root_path # image root path
        self.transform = transform
        self.genders = ["female", "male"]
        self.gender_map = {"man":"woman", "woman":"man", "male": "female", "female":"male"}

        self.all_occ_list = self.read_occ_file(occupation_path)

        self.data = []
        self.filenames = []
        for occ in self.all_occ_list:
            occ_name = "_".join(occ.split(" "))
            occ_dir = os.path.join(root_path, occ_name)
            for gender in self.genders:
                sub_dir = os.path.join(occ_dir, gender)
                for img_path in os.listdir(sub_dir):
                    self.data.append(os.path.join(sub_dir, img_path))
                    self.filenames.append(img_path)

    def __len__(self):
        return len(self.data)
    
    def read_occ_file(self, path):
        data = []
        with open(path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row['occupation'])
        return data

    def is_all_black(self, img):
        image = np.array(img)
        return np.all(image == [0, 0, 0])

    def __getitem__(self, idx):
        img_path = self.data[idx]
        img = Image.open(img_path).convert("RGB")
        nsfw_label = self.is_all_black(img)
        img, _ = self.transform(img, None)
        return self.filenames[idx], img, nsfw_label, img_path




    



def inverse_normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    tensor = tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor
    for i in range(tensor.shape[0]):
        tensor[i] = (tensor[i] * std[i]) + mean[i]
    return tensor

def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

class Bleacher():



        






    
if __name__=="__main__":
    occ_path = "../resources/occ_us.csv"
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=1)
    parser.add_argument("--log_file", type=str, default='log.txt')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--base_root", type=str, default="")
    parser.add_argument("--cf_root", type=str, default="")
    parser.add_argument("--occ_path", type=str, default=occ_path)
    parser.add_argument("--config", type=str, default="GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--grounded_checkpoint", type=str, default="groundingdino_swint_ogc.pth")
    parser.add_argument("--detect_prompt", type=str, default="a person.")
    parser.add_argument("--box_threshold", type=float, default=0.3)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--sub_exp", type=str, default="xl_bleach_person")
    parser.add_argument("--sam_checkpoint", type=str, default="sam_vit_h_4b8939.pth")
    args = parser.parse_args()
    image_filter = Bleacher(args)
    image_filter.main_bleach()