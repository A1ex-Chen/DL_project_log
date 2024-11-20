def __init__(self, data_root: str, data_src: str, training: bool,
    rand_state: int=0, summary: bool=True, int_dtype: type=np.int8,
    float_dtype: type=np.float16, output_dtype: type=np.float32,
    grth_scaling: str='none', dscptr_scaling: str='std', rnaseq_scaling:
    str='std', dscptr_nan_threshold: float=0.0, rnaseq_feature_usage: str=
    'source_scale', drug_feature_usage: str='both', validation_ratio: float
    =0.2, disjoint_drugs: bool=True, disjoint_cells: bool=True):
    """dataset = DrugRespDataset('./data/', 'NCI60', True)

        Construct a new drug response dataset based on the parameters
        provided. The process includes:
            * Downloading source data files;
            * Pre-processing (scaling, trimming, etc.);
            * Public attributes and other preparations.

        Args:
            data_root (str): path to data root folder.
            data_src (str): data source for drug response, must be one of
                'NCI60', 'CTRP', 'GDSC', 'CCLE', 'gCSI', and 'all'.
            training (bool): indicator for training.
            rand_state (int): random seed used for training/validation split
                and other processes that requires randomness.
            summary (bool): set True for printing dataset summary.

            int_dtype (type): integer dtype for data storage in RAM.
            float_dtype (type): float dtype for data storage in RAM.
            output_dtype (type): output dtype for neural network.

            grth_scaling (str): scaling method for drug response growth.
                Choose between 'none', 'std', and 'minmax'.
            dscptr_scaling (str): scaling method for drug descriptor.
                Choose between 'none', 'std', and 'minmax'.
            rnaseq_scaling (str): scaling method for RNA sequence (LINCS1K).
                Choose between 'none', 'std', and 'minmax'.
            dscptr_nan_threshold (float): NaN threshold for drug descriptor.
                If a column/feature or row/drug contains exceeding amount of
                NaN comparing to the threshold, the feature/drug will be
                dropped.

            rnaseq_feature_usage (str): RNA sequence usage. Choose between
                'combat', which is batch-effect-removed version of RNA
                sequence, or 'source_scale'.
            drug_feature_usage (str): drug feature usage. Choose between
                'fingerprint', 'descriptor', or 'both'.
            validation_ratio (float): portion of validation data out of all
                data samples. Note that this is not strictly the portion
                size. During the split, we will pick a percentage of
                drugs/cells and take the combination. The calculation will
                make sure that the expected validation size is accurate,
                but not strictly the case for a single random seed. Please
                refer to __split_drug_resp() for more details.
            disjoint_drugs (bool): indicator for disjoint drugs between
                training and validation dataset.
            disjoint_cells: indicator for disjoint cell lines between
                training and validation dataset.
        """
    self.__data_root = data_root
    self.data_source = data_src
    self.training = training
    self.__rand_state = rand_state
    self.__output_dtype = output_dtype
    if grth_scaling is None or grth_scaling == '':
        grth_scaling = 'none'
    grth_scaling = grth_scaling.lower()
    if dscptr_scaling is None or dscptr_scaling == '':
        dscptr_scaling = 'none'
    dscptr_scaling = dscptr_scaling
    if rnaseq_scaling is None or rnaseq_scaling == '':
        rnaseq_scaling = 'none'
    rnaseq_scaling = rnaseq_scaling
    self.__validation_ratio = validation_ratio
    self.__disjoint_drugs = disjoint_drugs
    self.__disjoint_cells = disjoint_cells
    self.__drug_resp_df = get_drug_resp_df(data_root=data_root,
        grth_scaling=grth_scaling, int_dtype=int_dtype, float_dtype=float_dtype
        )
    self.__drug_feature_df = get_drug_feature_df(data_root=data_root,
        drug_feature_usage=drug_feature_usage, dscptr_scaling=
        dscptr_scaling, dscptr_nan_thresh=dscptr_nan_threshold, int_dtype=
        int_dtype, float_dtype=float_dtype)
    self.__rnaseq_df = get_rna_seq_df(data_root=data_root,
        rnaseq_feature_usage=rnaseq_feature_usage, rnaseq_scaling=
        rnaseq_scaling, float_dtype=float_dtype)
    self.__split_drug_resp()
    self.drugs = self.__drug_resp_df['DRUG_ID'].unique().tolist()
    self.cells = self.__drug_resp_df['CELLNAME'].unique().tolist()
    self.num_records = len(self.__drug_resp_df)
    self.drug_feature_dim = self.__drug_feature_df.shape[1]
    self.rnaseq_dim = self.__rnaseq_df.shape[1]
    self.__drug_resp_array = self.__drug_resp_df.values
    self.__drug_feature_dict = {idx: row.values for idx, row in self.
        __drug_feature_df.iterrows()}
    self.__rnaseq_dict = {idx: row.values for idx, row in self.__rnaseq_df.
        iterrows()}
    self.__drug_resp_df = None
    self.__drug_feature_df = None
    self.__rnaseq_df = None
    if summary:
        print('=' * 80)
        print(('Training' if self.training else 'Validation') + 
            ' Drug Response Dataset Summary (Data Source: %6s):' % self.
            data_source)
        print('\t%i Drug Response Records .' % len(self.__drug_resp_array))
        print('\t%i Unique Drugs (feature dim: %4i).' % (len(self.drugs),
            self.drug_feature_dim))
        print('\t%i Unique Cell Lines (feature dim: %4i).' % (len(self.
            cells), self.rnaseq_dim))
        print('=' * 80)
