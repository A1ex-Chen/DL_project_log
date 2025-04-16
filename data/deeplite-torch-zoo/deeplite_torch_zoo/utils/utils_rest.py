# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

import time
import yaml
import inspect
import hashlib
import warnings
import functools
import os
import math
import contextlib
import subprocess
import logging.config

import pkg_resources as pkg

from pathlib import Path
from typing import Optional

from multiprocessing.pool import ThreadPool
from zipfile import is_zipfile, ZipFile
from tarfile import is_tarfile
from itertools import repeat

import numpy as np
import torch


KB_IN_MB_COUNT = 1024
LOGGING_NAME = 'deeplite-torch-zoo'
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))
TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'




set_logging(LOGGING_NAME)  # run before defining LOGGER
LOGGER = logging.getLogger(LOGGING_NAME)  # define globally






















class WorkingDirectory(contextlib.ContextDecorator):
    """Usage: @WorkingDirectory(dir) decorator or 'with WorkingDirectory(dir):' context manager."""

    def __init__(self, new_dir):
        """Sets the working directory to 'new_dir' upon instantiation."""
        self.dir = new_dir  # new dir
        self.cwd = Path.cwd().resolve()  # current dir

    def __enter__(self):
        """Changes the current directory to the specified directory."""
        os.chdir(self.dir)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore the current working directory on context exit."""
        os.chdir(self.cwd)






class Profile(contextlib.ContextDecorator):
    # YOLOv5 Profile class. Usage: @Profile() decorator or 'with Profile():' context manager
    def __init__(self, t=0.0):
        self.t = t
        self.cuda = torch.cuda.is_available()

    def __enter__(self):
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        self.dt = self.time() - self.start  # delta-time
        self.t += self.dt  # accumulate dt

    def time(self):
        if self.cuda:
            torch.cuda.synchronize()
        return time.time()









    return new_func


def print_args(args: Optional[dict] = None, show_file=True, show_func=False):
    # Print function arguments (optional args dict)
    x = inspect.currentframe().f_back  # previous frame
    file, _, func, _, _ = inspect.getframeinfo(x)
    if args is None:  # get args automatically
        args, _, _, frm = inspect.getargvalues(x)
        args = {k: v for k, v in frm.items() if k in args}
    file = Path(file).stem
    s = (f'{file}: ' if show_file else '') + (f'{func}: ' if show_func else '')
    kv = (', ').join(f'{k}={v}' for k, v in args.items())
    LOGGER.info("%s", colorstr(s) + kv)


def yaml_save(file='data.yaml', data={}):
    # Single-line safe yaml saving
    with open(file, 'w', encoding="utf-8") as f:
        yaml.safe_dump(
            {k: str(v) if isinstance(v, Path) else v for k, v in data.items()},
            f,
            sort_keys=False,
        )


def check_img_size(imgsz, s=32, floor=0):
    # Verify image size is a multiple of stride s in each dimension
    if isinstance(imgsz, int):  # integer i.e. img_size=640
        new_size = max(make_divisible(imgsz, int(s)), floor)
    else:  # list i.e. img_size=[640, 480]
        imgsz = list(imgsz)  # convert to list if tuple
        new_size = [max(make_divisible(x, int(s)), floor) for x in imgsz]
    if new_size != imgsz:
        LOGGER.warning(
            f'WARNING ⚠️ --img-size {imgsz} must be multiple of max stride {s}, updating to {new_size}'
        )
    return new_size


def file_size(path):
    # Return file/dir size (MB)
    mb = 1 << 20  # bytes to MiB (1024 ** 2)
    path = Path(path)
    if path.is_file():
        val = path.stat().st_size / mb
    elif path.is_dir():
        val = sum(f.stat().st_size for f in path.glob('**/*') if f.is_file()) / mb
    else:
        val = 0.0
    return val


def get_default_args(func):
    # Get func() default arguments
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def colorstr(*input):
    """Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')."""
    *args, string = (
        input if len(input) > 1 else ('blue', 'bold', input[0])
    )  # color arguments, string
    colors = {
        'black': '\033[30m',  # basic colors
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # bright colors
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # misc
        'bold': '\033[1m',
        'underline': '\033[4m',
    }
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    """
    Increments a file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.

    If the path exists and exist_ok is not set to True, the path will be incremented by appending a number and sep to
    the end of the path. If the path is a file, the file extension will be preserved. If the path is a directory, the
    number will be appended directly to the end of the path. If mkdir is set to True, the path will be created as a
    directory if it does not already exist.

    Args:
        path (str, pathlib.Path): Path to increment.
        exist_ok (bool, optional): If True, the path will not be incremented and returned as-is. Defaults to False.
        sep (str, optional): Separator to use between the path and the incrementation number. Defaults to ''.
        mkdir (bool, optional): Create a directory if it does not exist. Defaults to False.

    Returns:
        (pathlib.Path): Incremented path.
    """
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (
            (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        )

        # Method 1
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path


class WorkingDirectory(contextlib.ContextDecorator):
    """Usage: @WorkingDirectory(dir) decorator or 'with WorkingDirectory(dir):' context manager."""





def check_version(
    current: str = '0.0.0',
    minimum: str = '0.0.0',
    name: str = 'version ',
    pinned: bool = False,
    hard: bool = False,
    verbose: bool = False,
) -> bool:
    """
    Check current version against the required minimum version.

    Args:
        current (str): Current version.
        minimum (str): Required minimum version.
        name (str): Name to be used in warning message.
        pinned (bool): If True, versions must match exactly. If False, minimum version must be satisfied.
        hard (bool): If True, raise an AssertionError if the minimum version is not met.
        verbose (bool): If True, print warning message if minimum version is not met.

    Returns:
        (bool): True if minimum version is met, False otherwise.
    """
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    warning_message = f'WARNING ⚠️ {name}{minimum} is required by YOLOv8, but {name}{current} is currently installed'
    if hard:
        assert result, warning_message  # assert min requirements met
    if verbose and not result:
        LOGGER.warning(warning_message)
    return result


def is_dir_writeable(dir_path):
    """
    Check if a directory is writeable.

    Args:
        dir_path (str) or (Path): The path to the directory.

    Returns:
        bool: True if the directory is writeable, False otherwise.
    """
    return os.access(str(dir_path), os.W_OK)


class Profile(contextlib.ContextDecorator):
    # YOLOv5 Profile class. Usage: @Profile() decorator or 'with Profile():' context manager





def curl_download(url, filename, *, silent: bool = False) -> bool:
    """
    Download a file from a url to a filename using curl.
    """
    silent_option = 'sS' if silent else ''  # silent
    proc = subprocess.run(  # pylint: disable=subprocess-run-check
        [
            'curl',
            '-#',
            f'-{silent_option}L',
            url,
            '--output',
            filename,
            '--retry',
            '9',
            '-C',
            '-',
        ]
    )
    return proc.returncode == 0


def unzip_file(file, path=None, exclude=('.DS_Store', '__MACOSX')):
    # Unzip a *.zip file to path/, excluding files containing strings in exclude list
    if path is None:
        path = Path(file).parent  # default path
    with ZipFile(file) as zipObj:
        for f in zipObj.namelist():  # list all archived filenames in the zip
            if all(x not in f for x in exclude):
                zipObj.extract(f, path=path)


def download(url, dir='.', unzip=True, delete=True, curl=False, threads=1, retry=3):
    # Multithreaded file download and unzip function, used in data.yaml for autodownload

    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)  # make directory
    if threads > 1:
        pool = ThreadPool(threads)
        pool.imap(lambda x: download_one(*x), zip(url, repeat(dir)))  # multithreaded
        pool.close()
        pool.join()
    else:
        for u in [url] if isinstance(url, (str, Path)) else url:
            download_one(u, dir)


def get_pareto_set(variable1, variable2, ignore_indices=None):
    array = np.array([variable1, variable2]).T
    sorting_indices = array[:, 0].argsort()
    if ignore_indices is not None:
        sorting_indices = [idx for idx in sorting_indices if idx not in ignore_indices]

    # Sort on first dimension
    array = array[sorting_indices]
    ind_list = []

    # Add first row to pareto_frontier
    pareto_frontier = array[0:1, :]
    ind_list.append(sorting_indices[0])

    # Test next row against the last row in pareto_frontier
    for i, row in enumerate(array[1:, :]):
        if sum(row[x] >= pareto_frontier[-1][x] for x in range(len(row))) == len(row):
            # If it is better on all features add the row to pareto_frontier
            pareto_frontier = np.concatenate((pareto_frontier, [row]))
            ind_list.append(sorting_indices[i + 1])
    return ind_list