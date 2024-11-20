import math
import random
from models import VAE
from torch.utils.data import DataLoader

import flwr as fl
import torch
from utils import load_partition, Fl_Client, PARAMS




if __name__ == "__main__":
    # real situation
    main()