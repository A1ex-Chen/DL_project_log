"""
    File Name:          UnoPytorch/response_dataframes.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               9/4/18
    Python Version:     3.6.6
    File Description:
        This file takes care of all the dataframes related drug response.
"""


import logging
import multiprocessing
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from utils.data_processing.dataframe_scaling import scale_dataframe
from utils.data_processing.label_encoding import encode_label_to_int
from utils.miscellaneous.file_downloading import download_files

logger = logging.getLogger(__name__)

# Folders for raw/processed data
RAW_FOLDER = "./raw/"
PROC_FOLDER = "./processed/"

# All the filenames related to the drug response
DRUG_RESP_FILENAME = "rescaled_combined_single_drug_growth"









        num_cores = multiprocessing.cpu_count()
        combo_stats = Parallel(n_jobs=num_cores)(
            delayed(process_combo)(v) for _, v in combo_dict.items()
        )

        # Convert ths list of lists to dataframe
        cols = ["DRUG_ID", "CELLNAME", "NUM_REC", "AVG_GRTH", "CORR"]
        df = pd.DataFrame(combo_stats, columns=cols)

        # Convert data type into generic python types
        df[["NUM_REC"]] = df[["NUM_REC"]].astype(int)
        df[["AVG_GRTH", "CORR"]] = df[["AVG_GRTH", "CORR"]].astype(float)

        # save to disk for future usage
        try:
            os.makedirs(os.path.join(data_root, PROC_FOLDER))
        except FileExistsError:
            pass
        df.to_pickle(df_path)

    # Convert the dtypes for a more efficient, compact dataframe ##############
    df[["NUM_REC"]] = df[["NUM_REC"]].astype(int_dtype)
    df[["AVG_GRTH", "CORR"]] = df[["AVG_GRTH", "CORR"]].astype(float_dtype)
    return df


def get_drug_stats_df(
    data_root: str,
    grth_scaling: str,
    int_dtype: type = np.int16,
    float_dtype: type = np.float32,
):
    """df = get_drug_stats_df('./data/', 'std')

    This function loads the combination statistics file, iterates through
    all the drugs, and calculated the statistics including:
        * number of cell lines tested per drug;
        * number of drug response records per drug;
        * average of growth per drug;
        * average correlation of dose and growth per drug;
    for all the drugs.

    Note that if the dataframe is already stored in the processed folder,
    the function simply read from file and return after converting dtypes.

    Args:
        data_root (str): path to the data root folder.
        grth_scaling (str): scaling strategy for growth in drug response.
        int_dtype (type): int dtype for storage in RAM.
        float_dtype (float): int dtype for storage in RAM.

    Returns:
        pd.DataFrame: drug cell combination statistics dataframe, each row
            contains the following fields: ['DRUG_ID', 'NUM_CL', 'NUM_REC',
            'AVG_GRTH', 'AVG_CORR']
    """

    if int_dtype == np.int8:
        logger.warning("Integer length too smaller for drug statistics.")

    df_filename = "drug_stats_df(scaling=%s).pkl" % grth_scaling
    df_path = os.path.join(data_root, PROC_FOLDER, df_filename)

    # If the dataframe already exists, load and continue ######################
    if os.path.exists(df_path):
        df = pd.read_pickle(df_path)

    # Otherwise process combo statistics and save #############################
    else:
        logger.debug("Processing drug statics dataframe ... ")

        # Load combo (drug + cell) dataframe to construct drug statistics
        combo_stats_df = get_combo_stats_df(
            data_root=data_root,
            grth_scaling=grth_scaling,
            int_dtype=int,
            float_dtype=float,
        )

        # Using a dict to store all drug info with a single iteration
        drug_dict = {}

        # Each row in the combo stats dataframe contains:
        # ['DRUG_ID', 'CELLNAME','NUM_REC', 'AVG_GRTH', 'CORR']
        combo_stats_array = combo_stats_df.values
        for row in combo_stats_array:
            drug = row[0]
            if drug not in drug_dict:
                # Each dictionary value will be a list containing:
                # [num of cell, tuple of num of records,
                #  tuple of avg grth, tuple of corr]
                drug_dict[drug] = [0, (), (), ()]

            drug_dict[drug][0] += 1
            drug_dict[drug][1] += (row[2],)
            drug_dict[drug][2] += (row[3],)
            drug_dict[drug][3] += (row[4],)

        # Using list of lists (table) for much faster data access
        # This part is parallelized using joblib

        num_cores = multiprocessing.cpu_count()
        drug_stats = Parallel(n_jobs=num_cores)(
            delayed(process_drug)(k, v) for k, v in drug_dict.items()
        )

        # Convert ths list of lists to dataframe
        cols = ["DRUG_ID", "NUM_CL", "NUM_REC", "AVG_GRTH", "AVG_CORR"]
        df = pd.DataFrame(drug_stats, columns=cols)

        # Convert data type into generic python types
        df[["NUM_CL", "NUM_REC"]] = df[["NUM_CL", "NUM_REC"]].astype(int)
        df[["AVG_GRTH", "AVG_CORR"]] = df[["AVG_GRTH", "AVG_CORR"]].astype(float)
        df.set_index("DRUG_ID", inplace=True)

        # save to disk for future usage
        try:
            os.makedirs(os.path.join(data_root, PROC_FOLDER))
        except FileExistsError:
            pass
        df.to_pickle(df_path)

    # Convert the dtypes for a more efficient, compact dataframe ##############
    df[["NUM_CL", "NUM_REC"]] = df[["NUM_CL", "NUM_REC"]].astype(int_dtype)
    df[["AVG_GRTH", "AVG_CORR"]] = df[["AVG_GRTH", "AVG_CORR"]].astype(float_dtype)
    return df


def get_drug_anlys_df(data_root: str):
    """df = get_drug_anlys_df('./data/')

    This function will load the drug statistics dataframe and go on and
    classify all the drugs into 4 different categories:
        * high growth, high correlation
        * high growth, low correlation
        * low growth, high correlation
        * low growth, low correlation
    Using the median value of growth and correlation. The results will be
    returned as a dataframe with drug ID as index.

    Note that if the dataframe is already stored in the processed folder,
    the function simply read from file and return after converting dtypes.

    Args:
        data_root (str): path to the data root folder.

    Returns:
        pd.DataFrame: drug classes with growth and correlation, each row
            contains the following fields: ['HIGH_GROWTH', 'HIGH_CORR'],
            which are boolean features.
    """

    df_filename = "drug_anlys_df.pkl"
    df_path = os.path.join(data_root, PROC_FOLDER, df_filename)

    # If the dataframe already exists, load and return ########################
    if os.path.exists(df_path):
        return pd.read_pickle(df_path)

    # Otherwise process combo statistics and save #############################
    else:
        logger.debug("Processing drug analysis dataframe ... ")

        # Load drug statistics dataframe
        # Note that the scaling of growth has nothing to do with the analysis
        drug_stats_df = get_drug_stats_df(
            data_root=data_root, grth_scaling="none", int_dtype=int, float_dtype=float
        )

        drugs = drug_stats_df.index
        avg_grth = drug_stats_df["AVG_GRTH"].values
        avg_corr = drug_stats_df["AVG_CORR"].values

        high_grth = avg_grth > np.median(avg_grth)
        high_corr = avg_corr > np.median(avg_corr)

        drug_analysis_array = np.array([drugs, high_grth, high_corr]).transpose()

        # The returned dataframe will have two columns of boolean values,
        # indicating four different categories.
        df = pd.DataFrame(
            drug_analysis_array, columns=["DRUG_ID", "HIGH_GROWTH", "HIGH_CORR"]
        )
        df.set_index("DRUG_ID", inplace=True)

        # save to disk for future usage
        try:
            os.makedirs(os.path.join(data_root, PROC_FOLDER))
        except FileExistsError:
            pass
        df.to_pickle(df_path)

        return df


if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG)

    print("=" * 80 + "\nDrug response dataframe head:")
    print(get_drug_resp_df(data_root="../../data/", grth_scaling="none").head())
    # Test statistic data loading functions
    print("=" * 80 + "\nDrug analysis dataframe head:")
    print(get_drug_anlys_df(data_root="../../data/").head())

    # Plot histogram for drugs ('AVG_GRTH', 'AVG_CORR')
    get_drug_stats_df(data_root="../../data/", grth_scaling="none").hist(
        column=["AVG_GRTH", "AVG_CORR"], figsize=(16, 9), bins=20
    )

    plt.suptitle(
        "Histogram of average growth and average correlation between "
        "concentration and growth of all drugs"
    )
    plt.show()