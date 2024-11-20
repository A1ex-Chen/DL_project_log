import yaml
import os
import datetime
import sys
import json
import torch
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
chef_dir = os.path.join(script_dir, '../../../src')
sys.path.append(chef_dir)
from dist import init_distributed_mode
from ChEF.evaluator import Evaluator, load_config, sample_dataset
from ChEF.models import get_model
from ChEF.scenario import dataset_dict
# from ChEF.scenario.utils import rand_acc


    

if __name__ == '__main__':
    dist_args = init_distributed_mode()
    main(dist_args)