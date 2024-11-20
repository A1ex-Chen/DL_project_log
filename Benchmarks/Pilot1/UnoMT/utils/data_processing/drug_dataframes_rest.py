"""
    File Name:          UnoPytorch/drug_dataframes.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               9/4/18
    Python Version:     3.6.6
    File Description:
        This file takes care of all the dataframes related drug features.
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

# All the filenames related to the drug features
ECFP_FILENAME = "pan_drugs_dragon7_ECFP.tsv"
PFP_FILENAME = "pan_drugs_dragon7_PFP.tsv"
DSCPTR_FILENAME = "pan_drugs_dragon7_descriptors.tsv"
# Drug property file. Does not exist on FTP server.
DRUG_PROP_FILENAME = "combined.panther.targets"

# Use only the following target families for classification
TGT_FAMS = [
    "transferase",
    "oxidoreductase",
    "signaling molecule",
    "nucleic acid binding",
    "enzyme modulator",
    "hydrolase",
    "receptor",
    "transporter",
    "transcription factor",
    "chaperone",
]














if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG)

    print("=" * 80 + "\nDrug feature dataframe head:")
    print(
        get_drug_feature_df(
            data_root="../../data/",
            drug_feature_usage="both",
            dscptr_scaling="std",
            dscptr_nan_thresh=0.0,
        ).head()
    )

    print("=" * 80 + "\nDrug target families dataframe head:")
    print(get_drug_target_df(data_root="../../data/").head())

    print("=" * 80 + "\nDrug target families dataframe head:")
    print(get_drug_qed_df(data_root="../../data/", qed_scaling="none").head())