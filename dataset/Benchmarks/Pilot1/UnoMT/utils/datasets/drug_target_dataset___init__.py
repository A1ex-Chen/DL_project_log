def __init__(self, data_root: str, training: bool, rand_state: int=0,
    summary: bool=True, int_dtype: type=np.int8, float_dtype: type=np.
    float16, output_dtype: type=np.float32, dscptr_scaling: str='std',
    dscptr_nan_threshold: float=0.0, drug_feature_usage: str='both',
    validation_ratio: float=0.2):
    self.training = training
    self.__rand_state = rand_state
    self.__output_dtype = output_dtype
    self.__validation_ratio = validation_ratio
    self.__drug_feature_df = get_drug_feature_df(data_root=data_root,
        drug_feature_usage=drug_feature_usage, dscptr_scaling=
        dscptr_scaling, dscptr_nan_thresh=dscptr_nan_threshold, int_dtype=
        int_dtype, float_dtype=float_dtype)
    self.__drug_target_df = get_drug_target_df(data_root=data_root,
        int_dtype=int_dtype)
    self.__drug_feature_df['feature'] = list(map(float_dtype, self.
        __drug_feature_df.values.tolist()))
    self.__drug_target_df = pd.concat([self.__drug_target_df, self.
        __drug_feature_df[['feature']]], axis=1, join='inner')
    self.__split_drug_resp()
    self.drugs = self.__drug_target_df.index.tolist()
    self.num_drugs = len(self.drugs)
    self.drug_feature_dim = self.__drug_feature_df.shape[1]
    assert self.num_drugs == len(self.__drug_target_df)
    self.__drug_target_array = self.__drug_target_df.values
    self.__drug_feature_df = None
    self.__drug_target_df = None
    if summary:
        print('=' * 80)
        print(('Training' if self.training else 'Validation') +
            ' Drug Target Family Classification Dataset Summary:')
        print('\t%i Unique Drugs (feature dim: %4i).' % (self.num_drugs,
            self.drug_feature_dim))
        print('=' * 80)
