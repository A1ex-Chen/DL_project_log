from __future__ import absolute_import

import collections
import logging
import os
import sys
import threading
from itertools import cycle, islice

import numpy as np
import pandas as pd

try:
    from sklearn.impute import SimpleImputer as Imputer
except ImportError:
    from sklearn.preprocessing import Imputer

from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler

file_path = os.path.dirname(os.path.realpath(__file__))

import candle

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

# Number of data generator workers
WORKERS = 1


class BenchmarkP1B3(candle.Benchmark):
    def set_locals(self):
        """Functionality to set variables specific for the benchmark
        - required: set of required parameters for the benchmark.
        - additional_definitions: list of dictionaries describing the additional parameters for the
        benchmark.
        """

        if required is not None:
            self.required = set(required)
        if additional_definitions is not None:
            self.additional_definitions = additional_definitions


additional_definitions = [
    # Feature selection
    {
        "name": "cell_features",
        "nargs": "+",
        "choices": ["expression", "mirna", "proteome", "all", "categorical"],
        "help": 'use one or more cell line feature sets: "expression", "mirna", "proteome", "all"; or use "categorical" for one-hot encoding of cell lines',
    },
    {
        "name": "drug_features",
        "nargs": "+",
        "choices": ["descriptors", "latent", "all", "noise"],
        "help": "use dragon7 descriptors, latent representations from Aspuru-Guzik's SMILES autoencoder, or both, or random features; 'descriptors','latent', 'all', 'noise'",
    },
    {
        "name": "cell_noise_sigma",
        "type": float,
        "help": "standard deviation of guassian noise to add to cell line features during training",
    },
    # Output selection
    {
        "name": "min_logconc",
        "type": float,
        "help": "min log concentration of dose response data to use: -3.0 to -7.0",
    },
    {
        "name": "max_logconc",
        "type": float,
        "help": "max log concentration of dose response data to use: -3.0 to -7.0",
    },
    {
        "name": "subsample",
        "choices": ["naive_balancing", "none"],
        "help": "dose response subsample strategy; 'none' or 'naive_balancing'",
    },
    {
        "name": "category_cutoffs",
        "nargs": "+",
        "type": float,
        "help": "list of growth cutoffs (between -1 and +1) seperating non-response and response categories",
    },
    # Sample data selection
    {
        "name": "test_cell_split",
        "type": float,
        "help": "cell lines to use in test; if None use predefined unseen cell lines instead of sampling cell lines used in training",
    },
    # Test random model
    {
        "name": "scramble",
        "type": candle.str2bool,
        "default": False,
        "help": "randomly shuffle dose response data",
    },
    {
        "name": "workers",
        "type": int,
        "default": WORKERS,
        "help": "number of data generator workers",
    },
]

required = [
    "activation",
    "batch_size",
    "batch_normalization",
    "category_cutoffs",
    "cell_features",
    "dropout",
    "drug_features",
    "epochs",
    "feature_subsample",
    "initialization",
    "learning_rate",
    "loss",
    "min_logconc",
    "max_logconc",
    "optimizer",
    "rng_seed",
    "scaling",
    "subsample",
    "test_cell_split",
    "val_split",
    "cell_noise_sigma",
]
























additional_definitions = [
    # Feature selection
    {
        "name": "cell_features",
        "nargs": "+",
        "choices": ["expression", "mirna", "proteome", "all", "categorical"],
        "help": 'use one or more cell line feature sets: "expression", "mirna", "proteome", "all"; or use "categorical" for one-hot encoding of cell lines',
    },
    {
        "name": "drug_features",
        "nargs": "+",
        "choices": ["descriptors", "latent", "all", "noise"],
        "help": "use dragon7 descriptors, latent representations from Aspuru-Guzik's SMILES autoencoder, or both, or random features; 'descriptors','latent', 'all', 'noise'",
    },
    {
        "name": "cell_noise_sigma",
        "type": float,
        "help": "standard deviation of guassian noise to add to cell line features during training",
    },
    # Output selection
    {
        "name": "min_logconc",
        "type": float,
        "help": "min log concentration of dose response data to use: -3.0 to -7.0",
    },
    {
        "name": "max_logconc",
        "type": float,
        "help": "max log concentration of dose response data to use: -3.0 to -7.0",
    },
    {
        "name": "subsample",
        "choices": ["naive_balancing", "none"],
        "help": "dose response subsample strategy; 'none' or 'naive_balancing'",
    },
    {
        "name": "category_cutoffs",
        "nargs": "+",
        "type": float,
        "help": "list of growth cutoffs (between -1 and +1) seperating non-response and response categories",
    },
    # Sample data selection
    {
        "name": "test_cell_split",
        "type": float,
        "help": "cell lines to use in test; if None use predefined unseen cell lines instead of sampling cell lines used in training",
    },
    # Test random model
    {
        "name": "scramble",
        "type": candle.str2bool,
        "default": False,
        "help": "randomly shuffle dose response data",
    },
    {
        "name": "workers",
        "type": int,
        "default": WORKERS,
        "help": "number of data generator workers",
    },
]

required = [
    "activation",
    "batch_size",
    "batch_normalization",
    "category_cutoffs",
    "cell_features",
    "dropout",
    "drug_features",
    "epochs",
    "feature_subsample",
    "initialization",
    "learning_rate",
    "loss",
    "min_logconc",
    "max_logconc",
    "optimizer",
    "rng_seed",
    "scaling",
    "subsample",
    "test_cell_split",
    "val_split",
    "cell_noise_sigma",
]


def check_params(fileParams):
    # Allow for either dense or convolutional layer specification
    # if none found exit
    try:
        fileParams["dense"]
    except KeyError:
        try:
            fileParams["conv"]
        except KeyError:
            print(
                "Error! No dense or conv layers specified. Wrong file !! ... exiting "
            )
            raise
        else:
            try:
                fileParams["pool"]
            except KeyError:
                fileParams["pool"] = None
                print("Warning ! No pooling specified after conv layer.")


def extension_from_parameters(params, framework):
    """Construct string for saving model with annotation of parameters"""
    ext = framework
    ext += ".A={}".format(params["activation"])
    ext += ".B={}".format(params["batch_size"])
    ext += ".D={}".format(params["dropout"])
    ext += ".E={}".format(params["epochs"])
    if params["feature_subsample"]:
        ext += ".F={}".format(params["feature_subsample"])
    if "cell_noise_sigma" in params:
        ext += ".N={}".format(params["cell_noise_sigma"])
    if "conv" in params:
        name = "LC" if "locally_connected" in params else "C"
        layer_list = list(range(0, len(params["conv"])))
        for layer, i in enumerate(layer_list):
            filters = params["conv"][i][0]
            filter_len = params["conv"][i][1]
            stride = params["conv"][i][2]
            if filters <= 0 or filter_len <= 0 or stride <= 0:
                break
            ext += ".{}{}={},{},{}".format(name, layer + 1, filters, filter_len, stride)
        if "pool" in params and params["conv"][0] and params["conv"][1]:
            ext += ".P={}".format(params["pool"])
    if "dense" in params:
        for i, n in enumerate(params["dense"]):
            if n:
                ext += ".D{}={}".format(i + 1, n)
    if params["batch_normalization"]:
        ext += ".BN"
    ext += ".S={}".format(params["scaling"])

    return ext


def scale(df, scaling=None):
    """Scale data included in pandas dataframe.

    Parameters
    ----------
    df : pandas dataframe
        dataframe to scale
    scaling : 'maxabs', 'minmax', 'std', or None, optional (default 'std')
        type of scaling to apply
    """

    if scaling is None or scaling.lower() == "none":
        return df

    df = df.dropna(axis=1, how="any")

    # Scaling data
    if scaling == "maxabs":
        # Normalizing -1 to 1
        scaler = MaxAbsScaler()
    elif scaling == "minmax":
        # Scaling to [0,1]
        scaler = MinMaxScaler()
    else:
        # Standard normalization
        scaler = StandardScaler()

    mat = df.as_matrix()
    mat = scaler.fit_transform(mat)

    df = pd.DataFrame(mat, columns=df.columns)

    return df


def impute_and_scale(df, scaling="std"):
    """Impute missing values with mean and scale data included in pandas dataframe.

    Parameters
    ----------
    df : pandas dataframe
        dataframe to impute and scale
    scaling : 'maxabs' [-1,1], 'minmax' [0,1], 'std', or None, optional (default 'std')
        type of scaling to apply
    """

    df = df.dropna(axis=1, how="all")

    imputer = Imputer(strategy="mean")
    mat = imputer.fit_transform(df)

    if scaling is None or scaling.lower() == "none":
        return pd.DataFrame(mat, columns=df.columns)

    if scaling == "maxabs":
        scaler = MaxAbsScaler()
    elif scaling == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    mat = scaler.fit_transform(mat)

    df = pd.DataFrame(mat, columns=df.columns)

    return df


def load_cellline_expressions(path, dtype, ncols=None, scaling="std"):
    """Load cell line expression data, sub-select columns of gene expression
        randomly if specificed, scale the selected data and return a
        pandas dataframe.

    Parameters
    ----------
    path: string
        path to 'RNA_5_Platform_Gene_Transcript_Averaged_intensities.transposed.txt'
    dtype: numpy type
        precision (data type) for reading float values
    ncols : int or None
        number of columns (gene expression) to randomly subselect (default None : use all data)
    scaling : 'maxabs' [-1,1], 'minmax' [0,1], 'std', or None, optional (default 'std')
        type of scaling to apply
    """

    df = pd.read_csv(path, sep="\t", engine="c", na_values=["na", "-", ""])

    df1 = df["CellLine"]
    df1 = df1.map(lambda x: x.replace(".", ":"))
    df1.name = "CELLNAME"

    df2 = df.drop("CellLine", axis=1)

    total = df2.shape[1]
    if ncols and ncols < total:
        usecols = np.random.choice(total, size=ncols, replace=False)
        df2 = df2.iloc[:, usecols]

    df2 = impute_and_scale(df2, scaling)
    df2 = df2.astype(dtype)
    df = pd.concat([df1, df2], axis=1)

    return df


def load_cellline_mirna(path, dtype, ncols=None, scaling="std"):
    """Load cell line microRNA data, sub-select columns randomly if
        specificed, scale the selected data and return a pandas
        dataframe.

    Parameters
    ----------
    path: string
        path to 'RNA__microRNA_OSU_V3_chip_log2.transposed.txt'
    dtype: numpy type
        precision (data type) for reading float values
    ncols : int or None
        number of columns to randomly subselect (default None : use all data)
    scaling : 'maxabs' [-1,1], 'minmax' [0,1], 'std', or None, optional (default 'std')
        type of scaling to apply

    """

    df = pd.read_csv(path, sep="\t", engine="c", na_values=["na", "-", ""])

    df1 = df["CellLine"]
    df1 = df1.map(lambda x: x.replace(".", ":"))
    df1.name = "CELLNAME"

    df2 = df.drop("CellLine", axis=1)

    total = df2.shape[1]
    if ncols and ncols < total:
        usecols = np.random.choice(total, size=ncols, replace=False)
        df2 = df2.iloc[:, usecols]

    df2 = impute_and_scale(df2, scaling)
    df2 = df2.astype(dtype)
    df = pd.concat([df1, df2], axis=1)

    return df


def load_cellline_proteome(path, dtype, kinome_path=None, ncols=None, scaling="std"):
    """Load cell line microRNA data, sub-select columns randomly if
        specificed, scale the selected data and return a pandas
        dataframe.

    Parameters
    ----------
    path: string
        path to 'nci60_proteome_log2.transposed.tsv'
    dtype: numpy type
        precision (data type) for reading float values
    kinome_path: string or None (default None)
        path to 'nci60_kinome_log2.transposed.tsv'
    ncols : int or None
        number of columns to randomly subselect (default None : use all data)
    scaling : 'maxabs' [-1,1], 'minmax' [0,1], 'std', or None, optional (default 'std')
        type of scaling to apply

    """

    df = pd.read_csv(path, sep="\t", engine="c")
    df = df.set_index("CellLine")

    if kinome_path:
        df_k = pd.read_csv(kinome_path, sep="\t", engine="c")
        df_k = df_k.set_index("CellLine")
        df_k = df_k.add_suffix(".K")
        df = df.merge(df_k, left_index=True, right_index=True)

    index = df.index.map(lambda x: x.replace(".", ":"))

    total = df.shape[1]
    if ncols and ncols < total:
        usecols = np.random.choice(total, size=ncols, replace=False)
        df = df.iloc[:, usecols]

    df = impute_and_scale(df, scaling)
    df = df.astype(dtype)

    df.index = index
    df.index.names = ["CELLNAME"]
    df = df.reset_index()

    return df


def load_drug_descriptors(path, dtype, ncols=None, scaling="std"):
    """Load drug descriptor data, sub-select columns of drugs descriptors
        randomly if specificed, impute and scale the selected data, and return a
        pandas dataframe.

    Parameters
    ----------
    path: string
        path to 'descriptors.2D-NSC.5dose.filtered.txt'
    dtype: numpy type
        precision (data type) for reading float values
    ncols : int or None
        number of columns (drugs descriptors) to randomly subselect (default None : use all data)
    scaling : 'maxabs' [-1,1], 'minmax' [0,1], 'std', or None, optional (default 'std')
        type of scaling to apply
    """

    df = pd.read_csv(
        path,
        sep="\t",
        engine="c",
        na_values=["na", "-", ""],
        dtype=dtype,
        converters={"NAME": str},
    )

    df1 = pd.DataFrame(df.loc[:, "NAME"])
    df1.rename(columns={"NAME": "NSC"}, inplace=True)

    df2 = df.drop("NAME", axis=1)

    # # Filter columns if requested

    total = df2.shape[1]
    if ncols and ncols < total:
        usecols = np.random.choice(total, size=ncols, replace=False)
        df2 = df2.iloc[:, usecols]

    df2 = impute_and_scale(df2, scaling)
    df2 = df2.astype(dtype)

    df_dg = pd.concat([df1, df2], axis=1)

    return df_dg


def load_drug_autoencoded(path, dtype, ncols=None, scaling="std"):
    """Load drug latent representation from autoencoder, sub-select
    columns of drugs randomly if specificed, impute and scale the
    selected data, and return a pandas dataframe.

    Parameters
    ----------
    path: string
        path to 'Aspuru-Guzik_NSC_latent_representation_292D.csv'
    dtype: numpy type
        precision (data type) for reading float values
    ncols : int or None
        number of columns (drug latent representations) to randomly subselect (default None : use all data)
    scaling : 'maxabs' [-1,1], 'minmax' [0,1], 'std', or None, optional (default 'std')
        type of scaling to apply

    """

    df = pd.read_csv(path, engine="c", converters={"NSC": str}, dtype=dtype)

    df1 = pd.DataFrame(df.loc[:, "NSC"])
    df2 = df.drop("NSC", axis=1)

    total = df2.shape[1]
    if ncols and ncols < total:
        usecols = np.random.choice(total, size=ncols, replace=False)
        df2 = df2.iloc[:, usecols]

    df2 = impute_and_scale(df2, scaling)
    df2 = df2.astype(dtype)

    df = pd.concat([df1, df2], axis=1)

    return df


def load_dose_response(
    path, seed, dtype, min_logconc=-5.0, max_logconc=-5.0, subsample=None
):
    """Load cell line response to different drug compounds, sub-select response for a specific
        drug log concentration range and return a pandas dataframe.

    Parameters
    ----------
    path: string
        path to 'NCI60_dose_response_with_missing_z5_avg.csv'
    seed: integer
        seed for random generation
    dtype: numpy type
        precision (data type) for reading float values
    min_logconc : -3, -4, -5, -6, -7, optional (default -5)
        min log concentration of drug to return cell line growth
    max_logconc : -3, -4, -5, -6, -7, optional (default -5)
        max log concentration of drug to return cell line growth
    subsample: None, 'naive_balancing' (default None)
        subsampling strategy to use to balance the data based on growth
    """

    df = pd.read_csv(
        path,
        sep=",",
        engine="c",
        na_values=["na", "-", ""],
        dtype={
            "NSC": object,
            "CELLNAME": str,
            "LOG_CONCENTRATION": dtype,
            "GROWTH": dtype,
        },
    )

    df = df[
        (df["LOG_CONCENTRATION"] >= min_logconc)
        & (df["LOG_CONCENTRATION"] <= max_logconc)
    ]

    df = df[["NSC", "CELLNAME", "GROWTH", "LOG_CONCENTRATION"]]

    if subsample and subsample == "naive_balancing":
        df1 = df[df["GROWTH"] <= 0]
        df2 = df[(df["GROWTH"] > 0) & (df["GROWTH"] < 50)].sample(
            frac=0.7, random_state=seed
        )
        df3 = df[(df["GROWTH"] >= 50) & (df["GROWTH"] <= 100)].sample(
            frac=0.18, random_state=seed
        )
        df4 = df[df["GROWTH"] > 100].sample(frac=0.01, random_state=seed)
        df = pd.concat([df1, df2, df3, df4])

    df = df.set_index(["NSC"])

    return df


def stage_data():
    server = "http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/P1B3/"

    cell_expr_path = candle.fetch_file(
        server + "P1B3_cellline_expressions.tsv", "Pilot1", unpack=False
    )
    cell_mrna_path = candle.fetch_file(
        server + "P1B3_cellline_mirna.tsv", "Pilot1", unpack=False
    )
    cell_prot_path = candle.fetch_file(
        server + "P1B3_cellline_proteome.tsv", "Pilot1", unpack=False
    )
    cell_kino_path = candle.fetch_file(
        server + "P1B3_cellline_kinome.tsv", "Pilot1", unpack=False
    )
    drug_desc_path = candle.fetch_file(
        server + "P1B3_drug_descriptors.tsv", "Pilot1", unpack=False
    )
    drug_auen_path = candle.fetch_file(
        server + "P1B3_drug_latent.csv", "Pilot1", unpack=False
    )
    dose_resp_path = candle.fetch_file(
        server + "P1B3_dose_response.csv", "Pilot1", unpack=False
    )
    test_cell_path = candle.fetch_file(
        server + "P1B3_test_celllines.txt", "Pilot1", unpack=False
    )
    test_drug_path = candle.fetch_file(
        server + "P1B3_test_drugs.txt", "Pilot1", unpack=False
    )

    return (
        cell_expr_path,
        cell_mrna_path,
        cell_prot_path,
        cell_kino_path,
        drug_desc_path,
        drug_auen_path,
        dose_resp_path,
        test_cell_path,
        test_drug_path,
    )


class DataLoader(object):
    """Load merged drug response, drug descriptors and cell line essay data"""



class DataGenerator(object):
    """Generate training, validation or testing batches from loaded data"""

