import os
import sys
import yaml
import argparse
script_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(script_path))
sys.path.append(parent_dir)
import json
from metric import build_metric





if __name__ == '__main__':
    main()