import os
import glob
import random
import logging
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms
from .dataset import DataField
from DPT.dpt.transforms import Resize, NormalizeImage, PrepareForNet
logger = logging.getLogger(__name__)



class ResizeImage_mvs(object):
    def __init__(self):
        net_w = net_h = 384
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.transform = transforms.Compose(
                [
                    Resize(
                        net_w,
                        net_h,
                        resize_target=True,
                        keep_aspect_ratio=True,
                        ensure_multiple_of=32,
                        resize_method="minimal"
                    ),
                    normalization,
                    PrepareForNet(),
                ]
            )
    def __call__(self, img):
        img = self.transform(img)
        return img




class OurDataset(data.Dataset):
    '''Dataset class
    '''

    def __init__(self,  fields, n_views=0, mode='train'):
        # Attributes
        self.fields = fields
        print(mode,': ', n_views, ' views') 
        self.n_views = n_views

    def __len__(self):
        ''' Returns the length of the dataset.
        '''
        return self.n_views

    def __getitem__(self, idx):
        ''' Returns an item of the dataset.

        Args:
            idx (int): ID of data point
        '''
        data = {}
        for field_name, field in self.fields.items():
            field_data = field.load(idx)

            if isinstance(field_data, dict):
                for k, v in field_data.items():
                    if k is None:
                        data[field_name] = v
                    else:
                        data['%s.%s' % (field_name, k)] = v
            else:
                data[field_name] = field_data

        return data









class OurDataset(data.Dataset):
    '''Dataset class
    '''






def collate_remove_none(batch):
    ''' Collater that puts each data field into a tensor with outer dimension
        batch size.

    Args:
        batch: batch
    '''
    batch = list(filter(lambda x: x is not None, batch))
    return data.dataloader.default_collate(batch)


def worker_init_fn(worker_id):
    ''' Worker init function to ensure true randomness.
    '''
    random_data = os.urandom(4)
    base_seed = int.from_bytes(random_data, byteorder="big")
    np.random.seed(base_seed + worker_id)