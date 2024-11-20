import argparse
import torch

from deeplite_torch_zoo import get_model
from deeplite_torch_zoo.utils.profiler import profile_ram, ram_report






if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
