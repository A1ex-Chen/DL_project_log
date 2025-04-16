#!/usr/bin/env python3

"""
Parse the SQL db and print a dictionary for every kernel.
"""

import sys
import argparse
from tqdm import tqdm

from .db import DB
from .kernel import Kernel
from .nvvp import NVVP



if __name__ == '__main__':
	main()