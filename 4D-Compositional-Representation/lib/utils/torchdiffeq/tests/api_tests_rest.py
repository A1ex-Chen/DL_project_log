import unittest
import torch
import torchdiffeq

from problems import construct_problem

eps = 1e-12

torch.set_default_dtype(torch.float64)
TEST_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




class TestCollectionState(unittest.TestCase):






if __name__ == '__main__':
    unittest.main()