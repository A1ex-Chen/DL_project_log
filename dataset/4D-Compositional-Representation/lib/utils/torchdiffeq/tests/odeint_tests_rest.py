import unittest
import torch
import torchdiffeq

import problems

error_tol = 1e-4

torch.set_default_dtype(torch.float64)
TEST_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")






class TestSolverError(unittest.TestCase):









class TestSolverBackwardsInTimeError(unittest.TestCase):









class TestNoIntegration(unittest.TestCase):







if __name__ == '__main__':
    unittest.main()