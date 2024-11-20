def __init__(self, data_root: str, training: bool, rand_state: int=0,
    summary: bool=True, int_dtype: type=np.int8, float_dtype: type=np.
    float16, output_dtype: type=np.float32, rnaseq_scaling: str='std',
    rnaseq_feature_usage: str='source_scale', validation_ratio: float=0.2):
    """dataset = CLClassDataset('./data/', True)

        Construct a RNA sequence dataset based on the parameters provided.
        The process includes:
            * Downloading source data files;
            * Pre-processing (scaling);
            * Public attributes and other preparations.

        Args:
            data_root (str): path to data root folder.
            training (bool): indicator for training.
            rand_state (int): random seed used for training/validation split
                and other processes that requires randomness.
            summary (bool): set True for printing dataset summary.

            int_dtype (type): integer dtype for data storage in RAM.
            float_dtype (type): float dtype for data storage in RAM.
            output_dtype (type): output dtype for neural network.

            rnaseq_scaling (str): scaling method for RNA squence. Choose
                between 'none', 'std', and 'minmax'.
            rnaseq_feature_usage: RNA sequence data usage. Choose between
                'source_scale' and 'combat'.
            validation_ratio (float): portion of validation data out of all
                data samples.
        """
    self.__data_root = data_root
    self.training = training
    self.__rand_state = rand_state
    self.__output_dtype = output_dtype
    if rnaseq_scaling is None or rnaseq_scaling == '':
        rnaseq_scaling = 'none'
    self.__rnaseq_scaling = rnaseq_scaling.lower()
    self.__rnaseq_feature_usage = rnaseq_feature_usage
    self.__validation_ratio = validation_ratio
    self.__rnaseq_df = get_rna_seq_df(data_root=data_root,
        rnaseq_feature_usage=rnaseq_feature_usage, rnaseq_scaling=
        rnaseq_scaling, float_dtype=float_dtype)
    self.__cl_meta_df = get_cl_meta_df(data_root=data_root, int_dtype=int_dtype
        )
    self.__rnaseq_df['seq'] = list(map(float_dtype, self.__rnaseq_df.values
        .tolist()))
    self.__cl_df = pd.concat([self.__cl_meta_df, self.__rnaseq_df[['seq']]],
        axis=1, join='inner')
    num_data_src = len(get_label_dict(data_root, 'data_src_dict.txt'))
    enc_data_src = encode_int_to_onehot(self.__cl_df['data_src'].tolist(),
        num_classes=num_data_src)
    self.__cl_df['data_src'] = list(map(int_dtype, enc_data_src))
    self.__split_drug_resp()
    self.__cl_array = self.__cl_df.values
    self.cells = self.__cl_df.index.tolist()
    self.num_cells = self.__cl_df.shape[0]
    self.rnaseq_dim = len(self.__cl_df.iloc[0]['seq'])
    self.__rnaseq_df = None
    self.__cl_meta_df = None
    self.__cl_df = None
    if summary:
        print('=' * 80)
        print(('Training' if self.training else 'Validation') +
            ' RNA Sequence Dataset Summary:')
        print('\t%i Unique Cell Lines (feature dim: %4i).' % (self.
            num_cells, self.rnaseq_dim))
        print('=' * 80)
