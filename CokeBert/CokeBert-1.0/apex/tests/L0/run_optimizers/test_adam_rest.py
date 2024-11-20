import unittest
import os
import random

import torch
import apex

class TestFusedAdam(unittest.TestCase):









    @unittest.skip('Disable until 8/1/2019 adam/adamw upstream picked')

    @unittest.skip('No longer support fuse scaling')

    @unittest.skip('No longer support output fp16 param')



if __name__ == '__main__':
    script_path = os.path.dirname(os.path.realpath(__file__))
    unittest.main()