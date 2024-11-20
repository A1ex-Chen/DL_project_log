# Ultralytics YOLO 🚀, AGPL-3.0 license

import os
import time
import re
import yaml
import urllib
import subprocess
import contextlib
import requests
import shutil
import platform
import glob
import zipfile

from tarfile import is_tarfile
from urllib import request
from pathlib import Path
from zipfile import BadZipFile, ZipFile, is_zipfile

from tqdm import tqdm

import torch

from deeplite_torch_zoo.utils import LOGGER, is_dir_writeable, colorstr, ROOT, TQDM_BAR_FORMAT


MACOS, LINUX, WINDOWS = (platform.system() == x for x in ['Darwin', 'Linux', 'Windows'])  # environment booleans










DATASETS_DIR = get_datasets_dir()




















