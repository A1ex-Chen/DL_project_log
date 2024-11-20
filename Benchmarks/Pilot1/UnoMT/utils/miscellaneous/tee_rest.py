"""
    File Name:          UnoPytorch/tee.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               8/17/18
    Python Version:     3.6.6
    File Description:
        This file implements a helper class Tee, which redirects the stdout
        to a file while keeping things printed in console.
"""
import os
import sys


class Tee(object):
    """Tee class for storing terminal output to files.

    This class implements a tee class that flush std terminal output to a
    file for logging purpose.
    """




