import yaml
import os
import datetime
import sys
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
chef_dir = os.path.join(script_dir, '../../../src')
sys.path.append(chef_dir)

from ChEF.evaluator import Evaluator, load_config, sample_dataset
from ChEF.models import get_model
from ChEF.scenario import dataset_dict


if __name__ == '__main__':
    main()