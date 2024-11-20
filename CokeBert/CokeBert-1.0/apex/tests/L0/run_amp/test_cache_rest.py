import unittest

import functools as ft
import itertools as it

from apex import amp
from apex.amp import _amp_state
import torch
from torch import nn
import torch.nn.functional as F

from utils import common_init, HALF, FLOAT,\
    ALWAYS_HALF, ALWAYS_FLOAT, MATCH_INPUT


class WhitelistModule(torch.nn.Module):

    @staticmethod



class BlacklistModule(torch.nn.Module):

    @staticmethod



class PromoteModule(torch.nn.Module):

    @staticmethod


class TestCache(unittest.TestCase):


   
    # I could easily have these as a set of for loops in a single test,
    # instead of going for granularity.





        
        # Simulates first epoch
        training_step()
        
        # Simulates eval
        with torch.no_grad():
            loss = model(self.x).sum()
        
        # Simulates resuming training after eval
        training_step()

        _amp_state.handle._deactivate()
   
    # I could easily have these as a set of for loops in a single test,
    # instead of going for granularity.
    def test_whitelist_module_fp16_weight(self):
        self.train_eval_train_test(WhitelistModule, torch.float16)

    def test_whitelist_module_fp32_weight(self):
        self.train_eval_train_test(WhitelistModule, torch.float32)

    def test_blacklist_module_fp16_weight(self):
        self.train_eval_train_test(BlacklistModule, torch.float16)

    def test_blacklist_module_fp32_weight(self):
        self.train_eval_train_test(BlacklistModule, torch.float32)

    def test_promote_module_fp16_weight(self):
        self.train_eval_train_test(PromoteModule, torch.float16)

    def test_promote_module_fp32_weight(self):
        self.train_eval_train_test(PromoteModule, torch.float32)


if __name__ == '__main__':
    unittest.main()