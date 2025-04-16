import yaml
import os
import datetime
import sys
import json
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
chef_dir = os.path.join(script_dir, '../../../src')
sys.path.append(chef_dir)

from ChEF.evaluator import Evaluator, load_config, sample_dataset
from ChEF.models import get_model
from ChEF.scenario import dataset_dict
from ChEF.scenario.utils import rand_acc


    
    



if __name__ == '__main__':
    main()