import glob
import subprocess
import sys
from typing import List


sys.path.append(".")
from benchmark_text_to_image import ALL_T2I_CKPTS  # noqa: E402


PATTERN = "benchmark_*.py"


class SubprocessCallException(Exception):
    pass


# Taken from `test_examples_utils.py`




if __name__ == "__main__":
    main()