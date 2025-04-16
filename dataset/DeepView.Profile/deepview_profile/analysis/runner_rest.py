import argparse
import logging
import os
from deepview_profile.analysis.session import AnalysisSession
from deepview_profile.nvml import NVML
from deepview_profile.utils import release_memory
import weakref




if __name__ == "__main__":
    kwargs = {
        "format": "%(asctime)s %(levelname)-8s %(message)s",
        "datefmt": "%Y-%m-%d %H:%M",
        "level": logging.DEBUG,
    }
    logging.basicConfig(**kwargs)
    main()