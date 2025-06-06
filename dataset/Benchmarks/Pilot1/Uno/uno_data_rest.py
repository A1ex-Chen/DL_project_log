from __future__ import print_function

import collections
import json
import logging
import os
import pickle
from itertools import cycle, islice

import numpy as np
import pandas as pd
from tensorflow import keras

try:
    from sklearn.impute import SimpleImputer as Imputer
except ImportError:
    from sklearn.preprocessing import Imputer

from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler

file_path = os.path.dirname(os.path.realpath(__file__))

import candle

global_cache = {}

SEED = 2018

P1B3_URL = "http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/P1B3/"
DATA_URL = "http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/combo/"

logger = logging.getLogger(__name__)










































# def load_drug_smiles():
#     path = get_file(DATA_URL + 'ChemStructures_Consistent.smiles')

#     df = global_cache.get(path)
#     if df is None:
#         df = pd.read_csv(path, sep='\t', engine='c', dtype={'nsc_id':object})
#         df = df.rename(columns={'nsc_id': 'NSC'})
#         global_cache[path] = df

#     return df


















class CombinedDataLoader(object):
    def __init__(self, seed=SEED):
        self.seed = seed

    def load_from_cache(self, cache, params):
        """NOTE: How does this function return an error? (False?) -Wozniak"""
        param_fname = "{}.params.json".format(cache)
        if not os.path.isfile(param_fname):
            logger.warning("Cache parameter file does not exist: %s", param_fname)
            return False
        with open(param_fname) as param_file:
            try:
                cached_params = json.load(param_file)
            except json.JSONDecodeError as e:
                logger.warning("Could not decode parameter file %s: %s", param_fname, e)
                return False
        ignore_keys = ["cache", "partition_by", "single", "use_exported_data"]
        equal, diffs = dict_compare(params, cached_params, ignore_keys)
        if not equal:
            logger.warning(
                "Cache parameter mismatch: %s\nSaved: %s\nAttempted to load: %s",
                diffs,
                cached_params,
                params,
            )
            logger.warning("\nRemove %s to rebuild data cache.\n", param_fname)
            raise ValueError(
                "Could not load from a cache with incompatible keys:", diffs
            )
        else:
            fname = "{}.pkl".format(cache)
            if not os.path.isfile(fname):
                logger.warning("Cache file does not exist: %s", fname)
                return False
            with open(fname, "rb") as f:
                obj = pickle.load(f)
            self.__dict__.update(obj.__dict__)
            logger.info("Loaded data from cache: %s", fname)
            return True
        # NOTE: This is unreachable -Wozniak
        return False

    def save_to_cache(self, cache, params):
        for k in ["self", "cache", "single"]:
            if k in params:
                del params[k]
        dirname = os.path.dirname(cache)
        if dirname and not os.path.exists(dirname):
            logger.debug("Creating directory for cache: %s", dirname)
            os.makedirs(dirname)
        param_fname = "{}.params.json".format(cache)
        with open(param_fname, "w") as param_file:
            json.dump(params, param_file, sort_keys=True)
        fname = "{}.pkl".format(cache)
        with open(fname, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        logger.info("Saved data to cache: %s", fname)

    def partition_data(
        self,
        partition_by=None,
        cv_folds=1,
        train_split=0.7,
        val_split=0.2,
        cell_types=None,
        by_cell=None,
        by_drug=None,
        cell_subset_path=None,
        drug_subset_path=None,
        exclude_cells=[],
        exclude_drugs=[],
        exclude_indices=[],
    ):

        seed = self.seed
        train_sep_sources = self.train_sep_sources
        test_sep_sources = self.test_sep_sources
        df_response = self.df_response

        if not partition_by:
            if by_drug and by_cell:
                partition_by = "index"
            elif by_drug:
                partition_by = "cell"
            else:
                partition_by = "drug_pair"

        # Exclude specified cells / drugs / indices
        if exclude_cells != []:
            df_response = df_response[~df_response["Sample"].isin(exclude_cells)]
        if exclude_drugs != []:
            if np.isin("Drug", df_response.columns.values):
                df_response = df_response[~df_response["Drug1"].isin(exclude_drugs)]
            else:
                df_response = df_response[
                    ~df_response["Drug1"].isin(exclude_drugs)
                    & ~df_response["Drug2"].isin(exclude_drugs)
                ]
        if exclude_indices != []:
            df_response = df_response.drop(exclude_indices, axis=0)
            logger.info("Excluding indices specified")

        if partition_by != self.partition_by:
            df_response = df_response.assign(
                Group=assign_partition_groups(df_response, partition_by)
            )

        mask = df_response["Source"].isin(train_sep_sources)
        test_mask = df_response["Source"].isin(test_sep_sources)

        if by_drug:
            drug_ids = drug_name_to_ids(by_drug)
            logger.info("Mapped drug IDs for %s: %s", by_drug, drug_ids)
            mask &= (df_response["Drug1"].isin(drug_ids)) & (
                df_response["Drug2"].isnull()
            )
            test_mask &= (df_response["Drug1"].isin(drug_ids)) & (
                df_response["Drug2"].isnull()
            )

        if by_cell:
            cell_ids = cell_name_to_ids(by_cell)
            logger.info("Mapped sample IDs for %s: %s", by_cell, cell_ids)
            mask &= df_response["Sample"].isin(cell_ids)
            test_mask &= df_response["Sample"].isin(cell_ids)

        if cell_subset_path:
            cell_subset = read_set_from_file(cell_subset_path)
            mask &= df_response["Sample"].isin(cell_subset)
            test_mask &= df_response["Sample"].isin(cell_subset)

        if drug_subset_path:
            drug_subset = read_set_from_file(drug_subset_path)
            mask &= (df_response["Drug1"].isin(drug_subset)) & (
                (df_response["Drug2"].isnull())
                | (df_response["Drug2"].isin(drug_subset))
            )
            test_mask &= (df_response["Drug1"].isin(drug_subset)) & (
                (df_response["Drug2"].isnull())
                | (df_response["Drug2"].isin(drug_subset))
            )

        if cell_types:
            df_type = load_cell_metadata()
            cell_ids = set()
            for cell_type in cell_types:
                cells = df_type[
                    ~df_type["TUMOR_TYPE"].isnull()
                    & df_type["TUMOR_TYPE"].str.contains(cell_type, case=False)
                ]
                cell_ids |= set(cells["ANL_ID"].tolist())
                logger.info(
                    "Mapped sample tissue types for %s: %s",
                    cell_type,
                    set(cells["TUMOR_TYPE"].tolist()),
                )
            mask &= df_response["Sample"].isin(cell_ids)
            test_mask &= df_response["Sample"].isin(cell_ids)

        df_group = df_response[mask]["Group"].drop_duplicates().reset_index(drop=True)

        if cv_folds > 1:
            selector = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        else:
            selector = ShuffleSplit(
                n_splits=1,
                train_size=train_split,
                test_size=val_split,
                random_state=seed,
            )

        splits = selector.split(df_group)

        train_indexes = []
        val_indexes = []
        test_indexes = []

        for index, (train_group_index, val_group_index) in enumerate(splits):
            train_groups = set(df_group.values[train_group_index])
            val_groups = set(df_group.values[val_group_index])
            train_index = df_response.index[
                df_response["Group"].isin(train_groups) & mask
            ]
            val_index = df_response.index[df_response["Group"].isin(val_groups) & mask]
            test_index = df_response.index[
                ~df_response["Group"].isin(train_groups)
                & ~df_response["Group"].isin(val_groups)
                & test_mask
            ]

            train_indexes.append(train_index)
            val_indexes.append(val_index)
            test_indexes.append(test_index)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "CV fold %d: train data = %s, val data = %s, test data = %s",
                    index,
                    train_index.shape[0],
                    val_index.shape[0],
                    test_index.shape[0],
                )
                logger.debug(
                    "  train groups (%d): %s",
                    df_response.loc[train_index]["Group"].nunique(),
                    df_response.loc[train_index]["Group"].unique(),
                )
                logger.debug(
                    "  val groups ({%d}): %s",
                    df_response.loc[val_index]["Group"].nunique(),
                    df_response.loc[val_index]["Group"].unique(),
                )
                logger.debug(
                    "  test groups ({%d}): %s",
                    df_response.loc[test_index]["Group"].nunique(),
                    df_response.loc[test_index]["Group"].unique(),
                )

        self.partition_by = partition_by
        self.cv_folds = cv_folds
        self.train_indexes = train_indexes
        self.val_indexes = val_indexes
        self.test_indexes = test_indexes

    def build_feature_list(self, single=False):
        input_features = collections.OrderedDict()
        feature_shapes = collections.OrderedDict()

        if not self.agg_dose:
            doses = ["dose1", "dose2"] if not single else ["dose1"]
            for dose in doses:
                input_features[dose] = "dose"
                feature_shapes["dose"] = (1,)

        if self.encode_response_source:
            input_features["response.source"] = "response.source"
            feature_shapes["response.source"] = (self.df_source.shape[1] - 1,)

        for fea in self.cell_features:
            feature_type = "cell." + fea
            feature_name = "cell." + fea
            df_cell = getattr(self, self.cell_df_dict[fea])
            input_features[feature_name] = feature_type
            feature_shapes[feature_type] = (df_cell.shape[1] - 1,)

        drugs = ["drug1", "drug2"] if not single else ["drug1"]
        for drug in drugs:
            for fea in self.drug_features:
                feature_type = "drug." + fea
                feature_name = drug + "." + fea
                df_drug = getattr(self, self.drug_df_dict[fea])
                input_features[feature_name] = feature_type
                feature_shapes[feature_type] = (df_drug.shape[1] - 1,)

        input_dim = sum([np.prod(feature_shapes[x]) for x in input_features.values()])

        self.input_features = input_features
        self.feature_shapes = feature_shapes
        self.input_dim = input_dim

        logger.info("Input features shapes:")
        for k, v in self.input_features.items():
            logger.info("  {}: {}".format(k, self.feature_shapes[v]))
        logger.info("Total input dimensions: {}".format(self.input_dim))

    def load(
        self,
        cache=None,
        ncols=None,
        scaling="std",
        dropna=None,
        agg_dose=None,
        embed_feature_source=True,
        encode_response_source=True,
        cell_features=["rnaseq"],
        drug_features=["descriptors", "fingerprints"],
        cell_feature_subset_path=None,
        drug_feature_subset_path=None,
        drug_lower_response=1,
        drug_upper_response=-1,
        drug_response_span=0,
        drug_median_response_min=-1,
        drug_median_response_max=1,
        use_landmark_genes=False,
        use_filtered_genes=False,
        use_exported_data=None,
        preprocess_rnaseq=None,
        single=False,
        # train_sources=['GDSC', 'CTRP', 'ALMANAC', 'NCI60'],
        train_sources=["GDSC", "CTRP", "ALMANAC"],
        # val_sources='train',
        # test_sources=['CCLE', 'gCSI'],
        test_sources=["train"],
        partition_by="drug_pair",
    ):

        params = locals().copy()
        del params["self"]

        if not cell_features or "none" in [x.lower() for x in cell_features]:
            cell_features = []

        if not drug_features or "none" in [x.lower() for x in drug_features]:
            drug_features = []

        if cache and self.load_from_cache(cache, params):
            self.build_feature_list(single=single)
            return

        # rebuild cache equivalent from the exported dataset
        if use_exported_data is not None:
            with pd.HDFStore(use_exported_data, "r") as store:
                if "/model" in store.keys():
                    self.input_features = store.get_storer("model").attrs.input_features
                    self.feature_shapes = store.get_storer("model").attrs.feature_shapes
                    self.input_dim = sum(
                        [
                            np.prod(self.feature_shapes[x])
                            for x in self.input_features.values()
                        ]
                    )
                    self.test_sep_sources = []
                    return
                else:
                    logger.warning(
                        "\nExported dataset does not have model info. Please rebuild the dataset.\n"
                    )
                    raise ValueError(
                        "Could not load model info from the dataset:", use_exported_data
                    )

        logger.info("Loading data from scratch ...")

        if agg_dose:
            df_response = load_aggregated_single_response(
                target=agg_dose, combo_format=True
            )
        else:
            df_response = load_combined_dose_response()

        if logger.isEnabledFor(logging.INFO):
            logger.info("Summary of combined dose response by source:")
            logger.info(summarize_response_data(df_response, target=agg_dose))

        all_sources = df_response["Source"].unique()
        df_source = encode_sources(all_sources)

        if "all" in train_sources:
            train_sources = all_sources
        if "all" in test_sources:
            test_sources = all_sources
        elif "train" in test_sources:
            test_sources = train_sources

        train_sep_sources = [
            x for x in all_sources for y in train_sources if x.startswith(y)
        ]
        test_sep_sources = [
            x for x in all_sources for y in test_sources if x.startswith(y)
        ]

        ids1 = (
            df_response[["Drug1"]].drop_duplicates().rename(columns={"Drug1": "Drug"})
        )
        ids2 = (
            df_response[["Drug2"]].drop_duplicates().rename(columns={"Drug2": "Drug"})
        )
        df_drugs_with_response = (
            pd.concat([ids1, ids2]).drop_duplicates().dropna().reset_index(drop=True)
        )
        df_cells_with_response = (
            df_response[["Sample"]].drop_duplicates().reset_index(drop=True)
        )
        logger.info(
            "Combined raw dose response data has %d unique samples and %d unique drugs",
            df_cells_with_response.shape[0],
            df_drugs_with_response.shape[0],
        )

        if agg_dose:
            df_selected_drugs = None
        else:
            logger.info(
                "Limiting drugs to those with response min <= %g, max >= %g, span >= %g, median_min <= %g, median_max >= %g ...",
                drug_lower_response,
                drug_upper_response,
                drug_response_span,
                drug_median_response_min,
                drug_median_response_max,
            )
            df_selected_drugs = select_drugs_with_response_range(
                df_response,
                span=drug_response_span,
                lower=drug_lower_response,
                upper=drug_upper_response,
                lower_median=drug_median_response_min,
                upper_median=drug_median_response_max,
            )
            logger.info(
                "Selected %d drugs from %d",
                df_selected_drugs.shape[0],
                df_response["Drug1"].nunique(),
            )

        cell_feature_subset = read_set_from_file(cell_feature_subset_path)
        drug_feature_subset = read_set_from_file(drug_feature_subset_path)

        for fea in cell_features:
            fea = fea.lower()
            if fea == "rnaseq" or fea == "expression":
                df_cell_rnaseq = load_cell_rnaseq(
                    ncols=ncols,
                    scaling=scaling,
                    use_landmark_genes=use_landmark_genes,
                    use_filtered_genes=use_filtered_genes,
                    feature_subset=cell_feature_subset,
                    preprocess_rnaseq=preprocess_rnaseq,
                    embed_feature_source=embed_feature_source,
                )

        for fea in drug_features:
            fea = fea.lower()
            if fea == "descriptors":
                df_drug_desc = load_drug_descriptors(
                    ncols=ncols,
                    scaling=scaling,
                    dropna=dropna,
                    feature_subset=drug_feature_subset,
                )
            elif fea == "fingerprints":
                df_drug_fp = load_drug_fingerprints(
                    ncols=ncols,
                    scaling=scaling,
                    dropna=dropna,
                    feature_subset=drug_feature_subset,
                )
            elif fea == "mordred":
                df_drug_mordred = load_mordred_descriptors(
                    ncols=ncols,
                    scaling=scaling,
                    dropna=dropna,
                    feature_subset=drug_feature_subset,
                )

        # df_drug_desc, df_drug_fp = load_drug_data(ncols=ncols, scaling=scaling, dropna=dropna)

        cell_df_dict = {"rnaseq": "df_cell_rnaseq"}

        drug_df_dict = {
            "descriptors": "df_drug_desc",
            "fingerprints": "df_drug_fp",
            "mordred": "df_drug_mordred",
        }

        # df_cell_ids = df_cell_rnaseq[['Sample']].drop_duplicates()
        # df_drug_ids = pd.concat([df_drug_desc[['Drug']], df_drug_fp[['Drug']]]).drop_duplicates()

        logger.info("Filtering drug response data...")

        df_cell_ids = df_cells_with_response
        for fea in cell_features:
            df_cell = locals()[cell_df_dict[fea]]
            df_cell_ids = df_cell_ids.merge(df_cell[["Sample"]]).drop_duplicates()
        logger.info(
            "  %d molecular samples with feature and response data",
            df_cell_ids.shape[0],
        )

        df_drug_ids = df_drugs_with_response
        for fea in drug_features:
            df_drug = locals()[drug_df_dict[fea]]
            df_drug_ids = df_drug_ids.merge(df_drug[["Drug"]]).drop_duplicates()

        if df_selected_drugs is not None:
            df_drug_ids = df_drug_ids.merge(df_selected_drugs).drop_duplicates()
        logger.info(
            "  %d selected drugs with feature and response data", df_drug_ids.shape[0]
        )

        df_response = df_response[
            df_response["Sample"].isin(df_cell_ids["Sample"])
            & df_response["Drug1"].isin(df_drug_ids["Drug"])
            & (
                df_response["Drug2"].isin(df_drug_ids["Drug"])
                | df_response["Drug2"].isnull()
            )
        ]

        df_response = df_response[
            df_response["Source"].isin(train_sep_sources + test_sep_sources)
        ]

        df_response.reset_index(drop=True, inplace=True)

        if logger.isEnabledFor(logging.INFO):
            logger.info("Summary of filtered dose response by source:")
            logger.info(summarize_response_data(df_response, target=agg_dose))

        df_response = df_response.assign(
            Group=assign_partition_groups(df_response, partition_by)
        )

        self.agg_dose = agg_dose
        self.cell_features = cell_features
        self.drug_features = drug_features
        self.cell_df_dict = cell_df_dict
        self.drug_df_dict = drug_df_dict
        self.df_source = df_source
        self.df_response = df_response
        self.embed_feature_source = embed_feature_source
        self.encode_response_source = encode_response_source
        self.all_sources = all_sources
        self.train_sources = train_sources
        self.test_sources = test_sources
        self.train_sep_sources = train_sep_sources
        self.test_sep_sources = test_sep_sources
        self.partition_by = partition_by

        for var in list(drug_df_dict.values()) + list(cell_df_dict.values()):
            value = locals().get(var)
            if value is not None:
                setattr(self, var, value)

        self.build_feature_list(single=single)

        if cache:
            self.save_to_cache(cache, params)

    def get_cells_in_val(self):
        val_cell_ids = list(
            set(self.df_response.loc[self.val_indexes[0]]["Sample"].values)
        )
        return val_cell_ids

    def get_drugs_in_val(self):
        if np.isin("Drug", self.df_response.columns.values):
            val_drug_ids = list(
                set(self.df_response.loc[self.val_indexes[0]]["Drug"].values)
            )
        else:
            val_drug_ids = list(
                set(self.df_response.loc[self.val_indexes[0]]["Drug1"].values)
            )

        return val_drug_ids

    def get_index_in_val(self):
        val_indices = list(set(self.val_indexes[0]))

        return val_indices


class DataFeeder(keras.utils.Sequence):
    """Read from pre-joined dataset (HDF5 format) and feed data to the model."""

    def __init__(
        self,
        partition="train",
        filename=None,
        batch_size=32,
        shuffle=False,
        single=False,
        agg_dose=None,
        on_memory=False,
    ):
        self.partition = partition
        self.filename = filename
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.single = single
        self.agg_dose = agg_dose
        self.on_memory = on_memory
        self.target = agg_dose if agg_dose is not None else "Growth"

        self.store = pd.HDFStore(filename, mode="r")
        self.input_size = len(
            list(filter(lambda x: x.startswith("/x_train"), self.store.keys()))
        )
        try:
            y = self.store.select("y_{}".format(self.partition))
            self.index = y.index
            self.target_loc = y.columns.get_loc(self.target)
        except KeyError:
            self.index = []

        self.size = len(self.index)
        if self.size >= self.batch_size:
            self.steps = self.size // self.batch_size
        else:
            self.steps = 1
            self.batch_size = self.size
        self.index_map = np.arange(self.steps)
        if self.shuffle:
            np.random.shuffle(self.index_map)
        if self.on_memory:
            print("on_memory_loader activated")
            self.dataframe = {}
            current_partition_keys = list(
                map(
                    lambda x: x[1:],
                    filter(lambda x: self.partition in x, self.store.keys()),
                )
            )
            for key in current_partition_keys:
                self.dataframe[key] = self.store.get(key)
            self.store.close()

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        start = self.index_map[idx] * self.batch_size
        stop = (self.index_map[idx] + 1) * self.batch_size
        if self.on_memory:
            x = [
                self.dataframe["x_{0}_{1}".format(self.partition, i)].iloc[
                    start:stop, :
                ]
                for i in range(self.input_size)
            ]
            y = self.dataframe["y_{}".format(self.partition)].iloc[
                start:stop, self.target_loc
            ]
        else:
            x = [
                self.store.select(
                    "x_{0}_{1}".format(self.partition, i), start=start, stop=stop
                )
                for i in range(self.input_size)
            ]
            y = self.store.select(
                "y_{}".format(self.partition), start=start, stop=stop
            )[self.target]
        return x, y

    def reset(self):
        """empty method implementation to match reset() in CombinedDataGenerator"""
        pass

    def get_response(self, copy=False):
        if self.on_memory:
            return self._get_response_on_memory(copy=copy)
        else:
            return self._get_response_on_disk(copy=copy)

    def _get_response_on_memory(self, copy=False):
        if self.shuffle:
            self.index = [
                item
                for step in range(self.steps)
                for item in range(
                    self.index_map[step] * self.batch_size,
                    (self.index_map[step] + 1) * self.batch_size,
                )
            ]
            df = self.dataframe["y_{}".format(self.partition)].iloc[self.index, :]
        else:
            df = self.dataframe["y_{}".format(self.partition)]

        if self.agg_dose is None:
            df["Dose1"] = self.dataframe["x_{}_0".format(self.partition)].iloc[
                self.index, :
            ]
            if not self.single:
                df["Dose2"] = self.dataframe["x_{}_1".format(self.partition)].iloc[
                    self.index, :
                ]
        return df.copy() if copy else df

    def _get_response_on_disk(self, copy=False):
        if self.shuffle:
            self.index = [
                item
                for step in range(self.steps)
                for item in range(
                    self.index_map[step] * self.batch_size,
                    (self.index_map[step] + 1) * self.batch_size,
                )
            ]
            df = self.store.get("y_{}".format(self.partition)).iloc[self.index, :]
        else:
            df = self.store.get("y_{}".format(self.partition))

        if self.agg_dose is None:
            df["Dose1"] = self.store.get("x_{}_0".format(self.partition)).iloc[
                self.index, :
            ]
            if not self.single:
                df["Dose2"] = self.store.get("x_{}_1".format(self.partition)).iloc[
                    self.index, :
                ]
        return df.copy() if copy else df

    def close(self):
        if self.on_memory:
            pass
        else:
            self.store.close()


class CombinedDataGenerator(keras.utils.Sequence):
    """Generate training, validation or testing batches from loaded data"""

    def __init__(
        self,
        data,
        partition="train",
        fold=0,
        source=None,
        batch_size=32,
        shuffle=True,
        single=False,
        rank=0,
        total_ranks=1,
    ):
        self.data = data
        self.partition = partition
        self.batch_size = batch_size
        self.single = single

        if partition == "train":
            index = data.train_indexes[fold]
        elif partition == "val":
            index = data.val_indexes[fold]
        else:
            index = data.test_indexes[fold] if hasattr(data, "test_indexes") else []

        if source:
            df = data.df_response[["Source"]].iloc[index, :]
            index = df.index[df["Source"] == source]

        if shuffle:
            index = np.random.permutation(index)

        # sharing by rank
        samples_per_rank = len(index) // total_ranks
        samples_per_rank = self.batch_size * (samples_per_rank // self.batch_size)

        self.index = index[rank * samples_per_rank : (rank + 1) * samples_per_rank]
        self.index_cycle = cycle(self.index)
        self.size = len(self.index)
        self.steps = self.size // self.batch_size
        print(
            "partition:{0}, rank:{1}, sharded index size:{2}, batch_size:{3}, steps:{4}".format(
                partition, rank, self.size, self.batch_size, self.steps
            )
        )

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        shard = self.index[idx * self.batch_size : (idx + 1) * self.batch_size]
        x_list, y = self.get_slice(
            self.batch_size, single=self.single, partial_index=shard
        )
        return x_list, y

    def reset(self):
        self.index_cycle = cycle(self.index)

    def get_response(self, copy=False):
        df = self.data.df_response.iloc[self.index, :].drop(["Group"], axis=1)
        return df.copy() if copy else df

    def get_slice(
        self,
        size=None,
        contiguous=True,
        single=False,
        dataframe=False,
        partial_index=None,
    ):
        size = size or self.size
        single = single or self.data.agg_dose
        target = self.data.agg_dose or "Growth"

        if partial_index is not None:
            index = partial_index
        else:
            index = list(islice(self.index_cycle, size))
        df_orig = self.data.df_response.iloc[index, :]
        df = df_orig.copy()

        if not single:
            df["Swap"] = np.random.choice([True, False], df.shape[0])
            swap = df_orig["Drug2"].notnull() & df["Swap"]
            df.loc[swap, "Drug1"] = df_orig.loc[swap, "Drug2"]
            df.loc[swap, "Drug2"] = df_orig.loc[swap, "Drug1"]
            if not self.data.agg_dose:
                df["DoseSplit"] = np.random.uniform(0.001, 0.999, df.shape[0])
                df.loc[swap, "Dose1"] = df_orig.loc[swap, "Dose2"]
                df.loc[swap, "Dose2"] = df_orig.loc[swap, "Dose1"]

        split = df_orig["Drug2"].isnull()
        if not single:
            df.loc[split, "Drug2"] = df_orig.loc[split, "Drug1"]
            if not self.data.agg_dose:
                df.loc[split, "Dose1"] = df_orig.loc[split, "Dose1"] - np.log10(
                    df.loc[split, "DoseSplit"]
                )
                df.loc[split, "Dose2"] = df_orig.loc[split, "Dose1"] - np.log10(
                    1 - df.loc[split, "DoseSplit"]
                )

        if dataframe:
            cols = (
                [target, "Sample", "Drug1", "Drug2"]
                if not single
                else [target, "Sample", "Drug1"]
            )
            y = df[cols].reset_index(drop=True)
        else:
            y = values_or_dataframe(df[target], contiguous, dataframe)

        x_list = []

        if not self.data.agg_dose:
            doses = ["Dose1", "Dose2"] if not single else ["Dose1"]
            for dose in doses:
                x = values_or_dataframe(
                    df[[dose]].reset_index(drop=True), contiguous, dataframe
                )
                x_list.append(x)

        if self.data.encode_response_source:
            df_x = pd.merge(
                df[["Source"]], self.data.df_source, on="Source", how="left"
            )
            df_x.drop(["Source"], axis=1, inplace=True)
            x = values_or_dataframe(df_x, contiguous, dataframe)
            x_list.append(x)

        for fea in self.data.cell_features:
            df_cell = getattr(self.data, self.data.cell_df_dict[fea])
            df_x = pd.merge(df[["Sample"]], df_cell, on="Sample", how="left")
            df_x.drop(["Sample"], axis=1, inplace=True)
            x = values_or_dataframe(df_x, contiguous, dataframe)
            x_list.append(x)

        drugs = ["Drug1", "Drug2"] if not single else ["Drug1"]
        for drug in drugs:
            for fea in self.data.drug_features:
                df_drug = getattr(self.data, self.data.drug_df_dict[fea])
                df_x = pd.merge(
                    df[[drug]], df_drug, left_on=drug, right_on="Drug", how="left"
                )
                df_x.drop([drug, "Drug"], axis=1, inplace=True)
                if dataframe and not single:
                    df_x = df_x.add_prefix(drug + ".")
                x = values_or_dataframe(df_x, contiguous, dataframe)
                x_list.append(x)

        # print(x_list, y)
        return x_list, y

    def flow(self, single=False):
        while 1:
            x_list, y = self.get_slice(self.batch_size, single=single)
            yield x_list, y












class DataFeeder(keras.utils.Sequence):
    """Read from pre-joined dataset (HDF5 format) and feed data to the model."""










class CombinedDataGenerator(keras.utils.Sequence):
    """Generate training, validation or testing batches from loaded data"""








        usecols = [0] + [
            i for i, c in enumerate(df_cols.columns) if with_prefix(c) in feature_subset
        ]
        df_cols = df_cols.iloc[:, usecols]

    dtype_dict = dict((x, np.float32) for x in df_cols.columns[1:])
    df = pd.read_csv(path, engine="c", sep="\t", usecols=usecols, dtype=dtype_dict)
    if "Cancer_type_id" in df.columns:
        df.drop("Cancer_type_id", axis=1, inplace=True)

    prefixes = df["Sample"].str.extract("^([^.]*)", expand=False).rename("Source")
    sources = prefixes.drop_duplicates().reset_index(drop=True)
    df_source = pd.get_dummies(sources, prefix="rnaseq.source", prefix_sep=".")
    df_source = pd.concat([sources, df_source], axis=1)

    df1 = df["Sample"]
    if embed_feature_source:
        df_sample_source = pd.concat([df1, prefixes], axis=1)
        df1 = df_sample_source.merge(df_source, on="Source", how="left").drop(
            "Source", axis=1
        )
        logger.info(
            "Embedding RNAseq data source into features: %d additional columns",
            df1.shape[1] - 1,
        )

    df2 = df.drop("Sample", axis=1)
    if add_prefix:
        df2 = df2.add_prefix("rnaseq.")

    df2 = impute_and_scale(df2, scaling, imputing)

    df = pd.concat([df1, df2], axis=1)

    # scaling needs to be done before subsampling
    if sample_set:
        chosen = df["Sample"].str.startswith(sample_set)
        df = df[chosen].reset_index(drop=True)

    if index_by_sample:
        df = df.set_index("Sample")

    logger.info("Loaded combined RNAseq data: %s", df.shape)

    return df


def read_set_from_file(path):
    if path:
        with open(path, "r") as f:
            text = f.read().strip()
            subset = text.split()
    else:
        subset = None
    return subset


def select_drugs_with_response_range(
    df_response, lower=0, upper=0, span=0, lower_median=None, upper_median=None
):
    df = df_response.groupby(["Drug1", "Sample"])["Growth"].agg(
        ["min", "max", "median"]
    )
    df["span"] = df["max"].clip(lower=-1, upper=1) - df["min"].clip(lower=-1, upper=1)
    df = df.groupby("Drug1").mean().reset_index().rename(columns={"Drug1": "Drug"})
    mask = (df["min"] <= lower) & (df["max"] >= upper) & (df["span"] >= span)
    if lower_median:
        mask &= df["median"] >= lower_median
    if upper_median:
        mask &= df["median"] <= upper_median
    df_sub = df[mask]
    return df_sub


def summarize_response_data(df, target=None):
    target = target or "Growth"
    df_sum = df.groupby("Source").agg(
        {target: "count", "Sample": "nunique", "Drug1": "nunique", "Drug2": "nunique"}
    )
    if "Dose1" in df_sum:
        df_sum["MedianDose"] = df.groupby("Source").agg({"Dose1": "median"})
    return df_sum


def assign_partition_groups(df, partition_by="drug_pair"):
    if partition_by == "cell":
        group = df["Sample"]
    elif partition_by == "drug_pair":
        df_info = load_drug_info()
        id_dict = (
            df_info[["ID", "PUBCHEM"]]
            .drop_duplicates(["ID"])
            .set_index("ID")
            .iloc[:, 0]
        )
        group = df["Drug1"].copy()
        group[(df["Drug2"].notnull()) & (df["Drug1"] <= df["Drug2"])] = (
            df["Drug1"] + "," + df["Drug2"]
        )
        group[(df["Drug2"].notnull()) & (df["Drug1"] > df["Drug2"])] = (
            df["Drug2"] + "," + df["Drug1"]
        )
        group2 = group.map(id_dict)
        mapped = group2.notnull()
        group[mapped] = group2[mapped]
    elif partition_by == "index":
        group = df.reset_index()["index"]
    logger.info("Grouped response data by %s: %d groups", partition_by, group.nunique())
    return group


def dict_compare(d1, d2, ignore=[], expand=False):
    d1_keys = set(d1.keys()) - set(ignore)
    d2_keys = set(d2.keys()) - set(ignore)
    intersect_keys = d1_keys.intersection(d2_keys)
    added = d1_keys - d2_keys
    removed = d2_keys - d1_keys
    modified = set({x: (d1[x], d2[x]) for x in intersect_keys if d1[x] != d2[x]})
    common = set(x for x in intersect_keys if d1[x] == d2[x])
    equal = not (added or removed or modified)
    if expand:
        return equal, added, removed, modified, common
    else:
        return equal, added | removed | modified


def values_or_dataframe(df, contiguous=False, dataframe=False):
    if dataframe:
        return df
    mat = df.values
    if contiguous:
        mat = np.ascontiguousarray(mat)
    return mat


class CombinedDataLoader(object):
    def __init__(self, seed=SEED):
        self.seed = seed

    def load_from_cache(self, cache, params):
        """NOTE: How does this function return an error? (False?) -Wozniak"""
        param_fname = "{}.params.json".format(cache)
        if not os.path.isfile(param_fname):
            logger.warning("Cache parameter file does not exist: %s", param_fname)
            return False
        with open(param_fname) as param_file:
            try:
                cached_params = json.load(param_file)
            except json.JSONDecodeError as e:
                logger.warning("Could not decode parameter file %s: %s", param_fname, e)
                return False
        ignore_keys = ["cache", "partition_by", "single", "use_exported_data"]
        equal, diffs = dict_compare(params, cached_params, ignore_keys)
        if not equal:
            logger.warning(
                "Cache parameter mismatch: %s\nSaved: %s\nAttempted to load: %s",
                diffs,
                cached_params,
                params,
            )
            logger.warning("\nRemove %s to rebuild data cache.\n", param_fname)
            raise ValueError(
                "Could not load from a cache with incompatible keys:", diffs
            )
        else:
            fname = "{}.pkl".format(cache)
            if not os.path.isfile(fname):
                logger.warning("Cache file does not exist: %s", fname)
                return False
            with open(fname, "rb") as f:
                obj = pickle.load(f)
            self.__dict__.update(obj.__dict__)
            logger.info("Loaded data from cache: %s", fname)
            return True
        # NOTE: This is unreachable -Wozniak
        return False

    def save_to_cache(self, cache, params):
        for k in ["self", "cache", "single"]:
            if k in params:
                del params[k]
        dirname = os.path.dirname(cache)
        if dirname and not os.path.exists(dirname):
            logger.debug("Creating directory for cache: %s", dirname)
            os.makedirs(dirname)
        param_fname = "{}.params.json".format(cache)
        with open(param_fname, "w") as param_file:
            json.dump(params, param_file, sort_keys=True)
        fname = "{}.pkl".format(cache)
        with open(fname, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        logger.info("Saved data to cache: %s", fname)

    def partition_data(
        self,
        partition_by=None,
        cv_folds=1,
        train_split=0.7,
        val_split=0.2,
        cell_types=None,
        by_cell=None,
        by_drug=None,
        cell_subset_path=None,
        drug_subset_path=None,
        exclude_cells=[],
        exclude_drugs=[],
        exclude_indices=[],
    ):

        seed = self.seed
        train_sep_sources = self.train_sep_sources
        test_sep_sources = self.test_sep_sources
        df_response = self.df_response

        if not partition_by:
            if by_drug and by_cell:
                partition_by = "index"
            elif by_drug:
                partition_by = "cell"
            else:
                partition_by = "drug_pair"

        # Exclude specified cells / drugs / indices
        if exclude_cells != []:
            df_response = df_response[~df_response["Sample"].isin(exclude_cells)]
        if exclude_drugs != []:
            if np.isin("Drug", df_response.columns.values):
                df_response = df_response[~df_response["Drug1"].isin(exclude_drugs)]
            else:
                df_response = df_response[
                    ~df_response["Drug1"].isin(exclude_drugs)
                    & ~df_response["Drug2"].isin(exclude_drugs)
                ]
        if exclude_indices != []:
            df_response = df_response.drop(exclude_indices, axis=0)
            logger.info("Excluding indices specified")

        if partition_by != self.partition_by:
            df_response = df_response.assign(
                Group=assign_partition_groups(df_response, partition_by)
            )

        mask = df_response["Source"].isin(train_sep_sources)
        test_mask = df_response["Source"].isin(test_sep_sources)

        if by_drug:
            drug_ids = drug_name_to_ids(by_drug)
            logger.info("Mapped drug IDs for %s: %s", by_drug, drug_ids)
            mask &= (df_response["Drug1"].isin(drug_ids)) & (
                df_response["Drug2"].isnull()
            )
            test_mask &= (df_response["Drug1"].isin(drug_ids)) & (
                df_response["Drug2"].isnull()
            )

        if by_cell:
            cell_ids = cell_name_to_ids(by_cell)
            logger.info("Mapped sample IDs for %s: %s", by_cell, cell_ids)
            mask &= df_response["Sample"].isin(cell_ids)
            test_mask &= df_response["Sample"].isin(cell_ids)

        if cell_subset_path:
            cell_subset = read_set_from_file(cell_subset_path)
            mask &= df_response["Sample"].isin(cell_subset)
            test_mask &= df_response["Sample"].isin(cell_subset)

        if drug_subset_path:
            drug_subset = read_set_from_file(drug_subset_path)
            mask &= (df_response["Drug1"].isin(drug_subset)) & (
                (df_response["Drug2"].isnull())
                | (df_response["Drug2"].isin(drug_subset))
            )
            test_mask &= (df_response["Drug1"].isin(drug_subset)) & (
                (df_response["Drug2"].isnull())
                | (df_response["Drug2"].isin(drug_subset))
            )

        if cell_types:
            df_type = load_cell_metadata()
            cell_ids = set()
            for cell_type in cell_types:
                cells = df_type[
                    ~df_type["TUMOR_TYPE"].isnull()
                    & df_type["TUMOR_TYPE"].str.contains(cell_type, case=False)
                ]
                cell_ids |= set(cells["ANL_ID"].tolist())
                logger.info(
                    "Mapped sample tissue types for %s: %s",
                    cell_type,
                    set(cells["TUMOR_TYPE"].tolist()),
                )
            mask &= df_response["Sample"].isin(cell_ids)
            test_mask &= df_response["Sample"].isin(cell_ids)

        df_group = df_response[mask]["Group"].drop_duplicates().reset_index(drop=True)

        if cv_folds > 1:
            selector = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        else:
            selector = ShuffleSplit(
                n_splits=1,
                train_size=train_split,
                test_size=val_split,
                random_state=seed,
            )

        splits = selector.split(df_group)

        train_indexes = []
        val_indexes = []
        test_indexes = []

        for index, (train_group_index, val_group_index) in enumerate(splits):
            train_groups = set(df_group.values[train_group_index])
            val_groups = set(df_group.values[val_group_index])
            train_index = df_response.index[
                df_response["Group"].isin(train_groups) & mask
            ]
            val_index = df_response.index[df_response["Group"].isin(val_groups) & mask]
            test_index = df_response.index[
                ~df_response["Group"].isin(train_groups)
                & ~df_response["Group"].isin(val_groups)
                & test_mask
            ]

            train_indexes.append(train_index)
            val_indexes.append(val_index)
            test_indexes.append(test_index)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "CV fold %d: train data = %s, val data = %s, test data = %s",
                    index,
                    train_index.shape[0],
                    val_index.shape[0],
                    test_index.shape[0],
                )
                logger.debug(
                    "  train groups (%d): %s",
                    df_response.loc[train_index]["Group"].nunique(),
                    df_response.loc[train_index]["Group"].unique(),
                )
                logger.debug(
                    "  val groups ({%d}): %s",
                    df_response.loc[val_index]["Group"].nunique(),
                    df_response.loc[val_index]["Group"].unique(),
                )
                logger.debug(
                    "  test groups ({%d}): %s",
                    df_response.loc[test_index]["Group"].nunique(),
                    df_response.loc[test_index]["Group"].unique(),
                )

        self.partition_by = partition_by
        self.cv_folds = cv_folds
        self.train_indexes = train_indexes
        self.val_indexes = val_indexes
        self.test_indexes = test_indexes

    def build_feature_list(self, single=False):
        input_features = collections.OrderedDict()
        feature_shapes = collections.OrderedDict()

        if not self.agg_dose:
            doses = ["dose1", "dose2"] if not single else ["dose1"]
            for dose in doses:
                input_features[dose] = "dose"
                feature_shapes["dose"] = (1,)

        if self.encode_response_source:
            input_features["response.source"] = "response.source"
            feature_shapes["response.source"] = (self.df_source.shape[1] - 1,)

        for fea in self.cell_features:
            feature_type = "cell." + fea
            feature_name = "cell." + fea
            df_cell = getattr(self, self.cell_df_dict[fea])
            input_features[feature_name] = feature_type
            feature_shapes[feature_type] = (df_cell.shape[1] - 1,)

        drugs = ["drug1", "drug2"] if not single else ["drug1"]
        for drug in drugs:
            for fea in self.drug_features:
                feature_type = "drug." + fea
                feature_name = drug + "." + fea
                df_drug = getattr(self, self.drug_df_dict[fea])
                input_features[feature_name] = feature_type
                feature_shapes[feature_type] = (df_drug.shape[1] - 1,)

        input_dim = sum([np.prod(feature_shapes[x]) for x in input_features.values()])

        self.input_features = input_features
        self.feature_shapes = feature_shapes
        self.input_dim = input_dim

        logger.info("Input features shapes:")
        for k, v in self.input_features.items():
            logger.info("  {}: {}".format(k, self.feature_shapes[v]))
        logger.info("Total input dimensions: {}".format(self.input_dim))

    def load(
        self,
        cache=None,
        ncols=None,
        scaling="std",
        dropna=None,
        agg_dose=None,
        embed_feature_source=True,
        encode_response_source=True,
        cell_features=["rnaseq"],
        drug_features=["descriptors", "fingerprints"],
        cell_feature_subset_path=None,
        drug_feature_subset_path=None,
        drug_lower_response=1,
        drug_upper_response=-1,
        drug_response_span=0,
        drug_median_response_min=-1,
        drug_median_response_max=1,
        use_landmark_genes=False,
        use_filtered_genes=False,
        use_exported_data=None,
        preprocess_rnaseq=None,
        single=False,
        # train_sources=['GDSC', 'CTRP', 'ALMANAC', 'NCI60'],
        train_sources=["GDSC", "CTRP", "ALMANAC"],
        # val_sources='train',
        # test_sources=['CCLE', 'gCSI'],
        test_sources=["train"],
        partition_by="drug_pair",
    ):

        params = locals().copy()
        del params["self"]

        if not cell_features or "none" in [x.lower() for x in cell_features]:
            cell_features = []

        if not drug_features or "none" in [x.lower() for x in drug_features]:
            drug_features = []

        if cache and self.load_from_cache(cache, params):
            self.build_feature_list(single=single)
            return

        # rebuild cache equivalent from the exported dataset
        if use_exported_data is not None:
            with pd.HDFStore(use_exported_data, "r") as store:
                if "/model" in store.keys():
                    self.input_features = store.get_storer("model").attrs.input_features
                    self.feature_shapes = store.get_storer("model").attrs.feature_shapes
                    self.input_dim = sum(
                        [
                            np.prod(self.feature_shapes[x])
                            for x in self.input_features.values()
                        ]
                    )
                    self.test_sep_sources = []
                    return
                else:
                    logger.warning(
                        "\nExported dataset does not have model info. Please rebuild the dataset.\n"
                    )
                    raise ValueError(
                        "Could not load model info from the dataset:", use_exported_data
                    )

        logger.info("Loading data from scratch ...")

        if agg_dose:
            df_response = load_aggregated_single_response(
                target=agg_dose, combo_format=True
            )
        else:
            df_response = load_combined_dose_response()

        if logger.isEnabledFor(logging.INFO):
            logger.info("Summary of combined dose response by source:")
            logger.info(summarize_response_data(df_response, target=agg_dose))

        all_sources = df_response["Source"].unique()
        df_source = encode_sources(all_sources)

        if "all" in train_sources:
            train_sources = all_sources
        if "all" in test_sources:
            test_sources = all_sources
        elif "train" in test_sources:
            test_sources = train_sources

        train_sep_sources = [
            x for x in all_sources for y in train_sources if x.startswith(y)
        ]
        test_sep_sources = [
            x for x in all_sources for y in test_sources if x.startswith(y)
        ]

        ids1 = (
            df_response[["Drug1"]].drop_duplicates().rename(columns={"Drug1": "Drug"})
        )
        ids2 = (
            df_response[["Drug2"]].drop_duplicates().rename(columns={"Drug2": "Drug"})
        )
        df_drugs_with_response = (
            pd.concat([ids1, ids2]).drop_duplicates().dropna().reset_index(drop=True)
        )
        df_cells_with_response = (
            df_response[["Sample"]].drop_duplicates().reset_index(drop=True)
        )
        logger.info(
            "Combined raw dose response data has %d unique samples and %d unique drugs",
            df_cells_with_response.shape[0],
            df_drugs_with_response.shape[0],
        )

        if agg_dose:
            df_selected_drugs = None
        else:
            logger.info(
                "Limiting drugs to those with response min <= %g, max >= %g, span >= %g, median_min <= %g, median_max >= %g ...",
                drug_lower_response,
                drug_upper_response,
                drug_response_span,
                drug_median_response_min,
                drug_median_response_max,
            )
            df_selected_drugs = select_drugs_with_response_range(
                df_response,
                span=drug_response_span,
                lower=drug_lower_response,
                upper=drug_upper_response,
                lower_median=drug_median_response_min,
                upper_median=drug_median_response_max,
            )
            logger.info(
                "Selected %d drugs from %d",
                df_selected_drugs.shape[0],
                df_response["Drug1"].nunique(),
            )

        cell_feature_subset = read_set_from_file(cell_feature_subset_path)
        drug_feature_subset = read_set_from_file(drug_feature_subset_path)

        for fea in cell_features:
            fea = fea.lower()
            if fea == "rnaseq" or fea == "expression":
                df_cell_rnaseq = load_cell_rnaseq(
                    ncols=ncols,
                    scaling=scaling,
                    use_landmark_genes=use_landmark_genes,
                    use_filtered_genes=use_filtered_genes,
                    feature_subset=cell_feature_subset,
                    preprocess_rnaseq=preprocess_rnaseq,
                    embed_feature_source=embed_feature_source,
                )

        for fea in drug_features:
            fea = fea.lower()
            if fea == "descriptors":
                df_drug_desc = load_drug_descriptors(
                    ncols=ncols,
                    scaling=scaling,
                    dropna=dropna,
                    feature_subset=drug_feature_subset,
                )
            elif fea == "fingerprints":
                df_drug_fp = load_drug_fingerprints(
                    ncols=ncols,
                    scaling=scaling,
                    dropna=dropna,
                    feature_subset=drug_feature_subset,
                )
            elif fea == "mordred":
                df_drug_mordred = load_mordred_descriptors(
                    ncols=ncols,
                    scaling=scaling,
                    dropna=dropna,
                    feature_subset=drug_feature_subset,
                )

        # df_drug_desc, df_drug_fp = load_drug_data(ncols=ncols, scaling=scaling, dropna=dropna)

        cell_df_dict = {"rnaseq": "df_cell_rnaseq"}

        drug_df_dict = {
            "descriptors": "df_drug_desc",
            "fingerprints": "df_drug_fp",
            "mordred": "df_drug_mordred",
        }

        # df_cell_ids = df_cell_rnaseq[['Sample']].drop_duplicates()
        # df_drug_ids = pd.concat([df_drug_desc[['Drug']], df_drug_fp[['Drug']]]).drop_duplicates()

        logger.info("Filtering drug response data...")

        df_cell_ids = df_cells_with_response
        for fea in cell_features:
            df_cell = locals()[cell_df_dict[fea]]
            df_cell_ids = df_cell_ids.merge(df_cell[["Sample"]]).drop_duplicates()
        logger.info(
            "  %d molecular samples with feature and response data",
            df_cell_ids.shape[0],
        )

        df_drug_ids = df_drugs_with_response
        for fea in drug_features:
            df_drug = locals()[drug_df_dict[fea]]
            df_drug_ids = df_drug_ids.merge(df_drug[["Drug"]]).drop_duplicates()

        if df_selected_drugs is not None:
            df_drug_ids = df_drug_ids.merge(df_selected_drugs).drop_duplicates()
        logger.info(
            "  %d selected drugs with feature and response data", df_drug_ids.shape[0]
        )

        df_response = df_response[
            df_response["Sample"].isin(df_cell_ids["Sample"])
            & df_response["Drug1"].isin(df_drug_ids["Drug"])
            & (
                df_response["Drug2"].isin(df_drug_ids["Drug"])
                | df_response["Drug2"].isnull()
            )
        ]

        df_response = df_response[
            df_response["Source"].isin(train_sep_sources + test_sep_sources)
        ]

        df_response.reset_index(drop=True, inplace=True)

        if logger.isEnabledFor(logging.INFO):
            logger.info("Summary of filtered dose response by source:")
            logger.info(summarize_response_data(df_response, target=agg_dose))

        df_response = df_response.assign(
            Group=assign_partition_groups(df_response, partition_by)
        )

        self.agg_dose = agg_dose
        self.cell_features = cell_features
        self.drug_features = drug_features
        self.cell_df_dict = cell_df_dict
        self.drug_df_dict = drug_df_dict
        self.df_source = df_source
        self.df_response = df_response
        self.embed_feature_source = embed_feature_source
        self.encode_response_source = encode_response_source
        self.all_sources = all_sources
        self.train_sources = train_sources
        self.test_sources = test_sources
        self.train_sep_sources = train_sep_sources
        self.test_sep_sources = test_sep_sources
        self.partition_by = partition_by

        for var in list(drug_df_dict.values()) + list(cell_df_dict.values()):
            value = locals().get(var)
            if value is not None:
                setattr(self, var, value)

        self.build_feature_list(single=single)

        if cache:
            self.save_to_cache(cache, params)

    def get_cells_in_val(self):
        val_cell_ids = list(
            set(self.df_response.loc[self.val_indexes[0]]["Sample"].values)
        )
        return val_cell_ids

    def get_drugs_in_val(self):
        if np.isin("Drug", self.df_response.columns.values):
            val_drug_ids = list(
                set(self.df_response.loc[self.val_indexes[0]]["Drug"].values)
            )
        else:
            val_drug_ids = list(
                set(self.df_response.loc[self.val_indexes[0]]["Drug1"].values)
            )

        return val_drug_ids

    def get_index_in_val(self):
        val_indices = list(set(self.val_indexes[0]))

        return val_indices


class DataFeeder(keras.utils.Sequence):
    """Read from pre-joined dataset (HDF5 format) and feed data to the model."""

    def __init__(
        self,
        partition="train",
        filename=None,
        batch_size=32,
        shuffle=False,
        single=False,
        agg_dose=None,
        on_memory=False,
    ):
        self.partition = partition
        self.filename = filename
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.single = single
        self.agg_dose = agg_dose
        self.on_memory = on_memory
        self.target = agg_dose if agg_dose is not None else "Growth"

        self.store = pd.HDFStore(filename, mode="r")
        self.input_size = len(
            list(filter(lambda x: x.startswith("/x_train"), self.store.keys()))
        )
        try:
            y = self.store.select("y_{}".format(self.partition))
            self.index = y.index
            self.target_loc = y.columns.get_loc(self.target)
        except KeyError:
            self.index = []

        self.size = len(self.index)
        if self.size >= self.batch_size:
            self.steps = self.size // self.batch_size
        else:
            self.steps = 1
            self.batch_size = self.size
        self.index_map = np.arange(self.steps)
        if self.shuffle:
            np.random.shuffle(self.index_map)
        if self.on_memory:
            print("on_memory_loader activated")
            self.dataframe = {}
            current_partition_keys = list(
                map(
                    lambda x: x[1:],
                    filter(lambda x: self.partition in x, self.store.keys()),
                )
            )
            for key in current_partition_keys:
                self.dataframe[key] = self.store.get(key)
            self.store.close()

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        start = self.index_map[idx] * self.batch_size
        stop = (self.index_map[idx] + 1) * self.batch_size
        if self.on_memory:
            x = [
                self.dataframe["x_{0}_{1}".format(self.partition, i)].iloc[
                    start:stop, :
                ]
                for i in range(self.input_size)
            ]
            y = self.dataframe["y_{}".format(self.partition)].iloc[
                start:stop, self.target_loc
            ]
        else:
            x = [
                self.store.select(
                    "x_{0}_{1}".format(self.partition, i), start=start, stop=stop
                )
                for i in range(self.input_size)
            ]
            y = self.store.select(
                "y_{}".format(self.partition), start=start, stop=stop
            )[self.target]
        return x, y

    def reset(self):
        """empty method implementation to match reset() in CombinedDataGenerator"""
        pass

    def get_response(self, copy=False):
        if self.on_memory:
            return self._get_response_on_memory(copy=copy)
        else:
            return self._get_response_on_disk(copy=copy)

    def _get_response_on_memory(self, copy=False):
        if self.shuffle:
            self.index = [
                item
                for step in range(self.steps)
                for item in range(
                    self.index_map[step] * self.batch_size,
                    (self.index_map[step] + 1) * self.batch_size,
                )
            ]
            df = self.dataframe["y_{}".format(self.partition)].iloc[self.index, :]
        else:
            df = self.dataframe["y_{}".format(self.partition)]

        if self.agg_dose is None:
            df["Dose1"] = self.dataframe["x_{}_0".format(self.partition)].iloc[
                self.index, :
            ]
            if not self.single:
                df["Dose2"] = self.dataframe["x_{}_1".format(self.partition)].iloc[
                    self.index, :
                ]
        return df.copy() if copy else df

    def _get_response_on_disk(self, copy=False):
        if self.shuffle:
            self.index = [
                item
                for step in range(self.steps)
                for item in range(
                    self.index_map[step] * self.batch_size,
                    (self.index_map[step] + 1) * self.batch_size,
                )
            ]
            df = self.store.get("y_{}".format(self.partition)).iloc[self.index, :]
        else:
            df = self.store.get("y_{}".format(self.partition))

        if self.agg_dose is None:
            df["Dose1"] = self.store.get("x_{}_0".format(self.partition)).iloc[
                self.index, :
            ]
            if not self.single:
                df["Dose2"] = self.store.get("x_{}_1".format(self.partition)).iloc[
                    self.index, :
                ]
        return df.copy() if copy else df

    def close(self):
        if self.on_memory:
            pass
        else:
            self.store.close()


class CombinedDataGenerator(keras.utils.Sequence):
    """Generate training, validation or testing batches from loaded data"""

    def __init__(
        self,
        data,
        partition="train",
        fold=0,
        source=None,
        batch_size=32,
        shuffle=True,
        single=False,
        rank=0,
        total_ranks=1,
    ):
        self.data = data
        self.partition = partition
        self.batch_size = batch_size
        self.single = single

        if partition == "train":
            index = data.train_indexes[fold]
        elif partition == "val":
            index = data.val_indexes[fold]
        else:
            index = data.test_indexes[fold] if hasattr(data, "test_indexes") else []

        if source:
            df = data.df_response[["Source"]].iloc[index, :]
            index = df.index[df["Source"] == source]

        if shuffle:
            index = np.random.permutation(index)

        # sharing by rank
        samples_per_rank = len(index) // total_ranks
        samples_per_rank = self.batch_size * (samples_per_rank // self.batch_size)

        self.index = index[rank * samples_per_rank : (rank + 1) * samples_per_rank]
        self.index_cycle = cycle(self.index)
        self.size = len(self.index)
        self.steps = self.size // self.batch_size
        print(
            "partition:{0}, rank:{1}, sharded index size:{2}, batch_size:{3}, steps:{4}".format(
                partition, rank, self.size, self.batch_size, self.steps
            )
        )

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        shard = self.index[idx * self.batch_size : (idx + 1) * self.batch_size]
        x_list, y = self.get_slice(
            self.batch_size, single=self.single, partial_index=shard
        )
        return x_list, y

    def reset(self):
        self.index_cycle = cycle(self.index)

    def get_response(self, copy=False):
        df = self.data.df_response.iloc[self.index, :].drop(["Group"], axis=1)
        return df.copy() if copy else df

    def get_slice(
        self,
        size=None,
        contiguous=True,
        single=False,
        dataframe=False,
        partial_index=None,
    ):
        size = size or self.size
        single = single or self.data.agg_dose
        target = self.data.agg_dose or "Growth"

        if partial_index is not None:
            index = partial_index
        else:
            index = list(islice(self.index_cycle, size))
        df_orig = self.data.df_response.iloc[index, :]
        df = df_orig.copy()

        if not single:
            df["Swap"] = np.random.choice([True, False], df.shape[0])
            swap = df_orig["Drug2"].notnull() & df["Swap"]
            df.loc[swap, "Drug1"] = df_orig.loc[swap, "Drug2"]
            df.loc[swap, "Drug2"] = df_orig.loc[swap, "Drug1"]
            if not self.data.agg_dose:
                df["DoseSplit"] = np.random.uniform(0.001, 0.999, df.shape[0])
                df.loc[swap, "Dose1"] = df_orig.loc[swap, "Dose2"]
                df.loc[swap, "Dose2"] = df_orig.loc[swap, "Dose1"]

        split = df_orig["Drug2"].isnull()
        if not single:
            df.loc[split, "Drug2"] = df_orig.loc[split, "Drug1"]
            if not self.data.agg_dose:
                df.loc[split, "Dose1"] = df_orig.loc[split, "Dose1"] - np.log10(
                    df.loc[split, "DoseSplit"]
                )
                df.loc[split, "Dose2"] = df_orig.loc[split, "Dose1"] - np.log10(
                    1 - df.loc[split, "DoseSplit"]
                )

        if dataframe:
            cols = (
                [target, "Sample", "Drug1", "Drug2"]
                if not single
                else [target, "Sample", "Drug1"]
            )
            y = df[cols].reset_index(drop=True)
        else:
            y = values_or_dataframe(df[target], contiguous, dataframe)

        x_list = []

        if not self.data.agg_dose:
            doses = ["Dose1", "Dose2"] if not single else ["Dose1"]
            for dose in doses:
                x = values_or_dataframe(
                    df[[dose]].reset_index(drop=True), contiguous, dataframe
                )
                x_list.append(x)

        if self.data.encode_response_source:
            df_x = pd.merge(
                df[["Source"]], self.data.df_source, on="Source", how="left"
            )
            df_x.drop(["Source"], axis=1, inplace=True)
            x = values_or_dataframe(df_x, contiguous, dataframe)
            x_list.append(x)

        for fea in self.data.cell_features:
            df_cell = getattr(self.data, self.data.cell_df_dict[fea])
            df_x = pd.merge(df[["Sample"]], df_cell, on="Sample", how="left")
            df_x.drop(["Sample"], axis=1, inplace=True)
            x = values_or_dataframe(df_x, contiguous, dataframe)
            x_list.append(x)

        drugs = ["Drug1", "Drug2"] if not single else ["Drug1"]
        for drug in drugs:
            for fea in self.data.drug_features:
                df_drug = getattr(self.data, self.data.drug_df_dict[fea])
                df_x = pd.merge(
                    df[[drug]], df_drug, left_on=drug, right_on="Drug", how="left"
                )
                df_x.drop([drug, "Drug"], axis=1, inplace=True)
                if dataframe and not single:
                    df_x = df_x.add_prefix(drug + ".")
                x = values_or_dataframe(df_x, contiguous, dataframe)
                x_list.append(x)

        # print(x_list, y)
        return x_list, y

    def flow(self, single=False):
        while 1:
            x_list, y = self.get_slice(self.batch_size, single=single)
            yield x_list, y


def test_generator(loader):
    gen = CombinedDataGenerator(loader).flow()
    x_list, y = next(gen)
    print("x shapes:")
    for x in x_list:
        print(x.shape)
    print("y shape:")
    print(y.shape)