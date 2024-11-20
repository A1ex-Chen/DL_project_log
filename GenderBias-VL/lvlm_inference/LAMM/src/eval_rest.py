import os
import yaml
import torch
from dist import init_distributed_mode

from ChEF.models import get_model
from ChEF.scenario import dataset_dict
from ChEF.evaluator import Evaluator, load_config, sample_dataset



if __name__ == '__main__':
    dist_args = init_distributed_mode()
    main(dist_args)