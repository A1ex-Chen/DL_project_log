"""
    File Name:          UnoPytorch/cell_line_dataframes.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               9/4/18
    Python Version:     3.6.6
    File Description:
        This file takes care of all the dataframes related cell lines.
"""
import logging
import os

import numpy as np
import pandas as pd
from utils.data_processing.dataframe_scaling import scale_dataframe
from utils.data_processing.label_encoding import encode_label_to_int
from utils.miscellaneous.file_downloading import download_files

logger = logging.getLogger(__name__)

# Folders for raw/processed data
RAW_FOLDER = "./raw/"
PROC_FOLDER = "./processed/"

# All the filenames related to cell lines
CL_METADATA_FILENAME = "combined_cl_metadata"
RNASEQ_SOURCE_SCALE_FILENAME = "combined_rnaseq_data_lincs1000_source_scale"
RNASEQ_COMBAT_FILENAME = "combined_rnaseq_data_lincs1000_combat"






if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG)

    print("=" * 80 + "\nRNA sequence dataframe head:")
    print(
        get_rna_seq_df(
            data_root="../../data/",
            rnaseq_feature_usage="source_scale",
            rnaseq_scaling="std",
        ).head()
    )

    print("=" * 80 + "\nCell line metadata dataframe head:")
    print(get_cl_meta_df(data_root="../../data/").head())