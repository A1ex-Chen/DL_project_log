# from cgi import test
import torch

from options import args
from models import model_factory
from dataloaders import dataloader_factory
from trainers import trainer_factory
from utils import *






if __name__ == '__main__':
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        print('Test mode')
        test()
    else:
        raise ValueError('Invalid mode')