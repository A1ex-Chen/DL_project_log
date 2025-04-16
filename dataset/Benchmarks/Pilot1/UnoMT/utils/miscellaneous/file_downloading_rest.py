"""
    File Name:          UnoPytorch/file_downloading.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               8/13/18
    Python Version:     3.6.6
    File Description:

"""
import errno
import logging
import os
import urllib

FTP_ROOT = "http://ftp.mcs.anl.gov/pub/candle/public/" "benchmarks/Pilot1/combo/"

logger = logging.getLogger(__name__)

