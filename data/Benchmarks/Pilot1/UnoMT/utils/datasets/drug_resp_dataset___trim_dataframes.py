def __trim_dataframes(self):
    """self.__trim_dataframes(trim_data_source=True)

        This function trims three dataframes to make sure that drug response
        dataframe, RNA sequence dataframe, and drug feature dataframe are
        sharing the same list of cell lines and drugs.

        Returns:
            None
        """
    if self.data_source.lower() != 'all':
        logger.debug('Specifying data source %s ... ' % self.data_source)
        data_src_dict = get_label_dict(data_root=self.__data_root,
            dict_name='data_src_dict.txt')
        encoded_data_src = data_src_dict[self.data_source]
        self.__drug_resp_df = self.__drug_resp_df.loc[self.__drug_resp_df[
            'SOURCE'] == encoded_data_src]
    logger.debug('Trimming dataframes on common cell lines and drugs ... ')
    cell_set = set(self.__drug_resp_df['CELLNAME'].unique()) & set(self.
        __rnaseq_df.index.values)
    drug_set = set(self.__drug_resp_df['DRUG_ID'].unique()) & set(self.
        __drug_feature_df.index.values)
    self.__drug_resp_df = self.__drug_resp_df.loc[self.__drug_resp_df[
        'CELLNAME'].isin(cell_set) & self.__drug_resp_df['DRUG_ID'].isin(
        drug_set)]
    self.__rnaseq_df = self.__rnaseq_df[self.__rnaseq_df.index.isin(cell_set)]
    self.__drug_feature_df = self.__drug_feature_df[self.__drug_feature_df.
        index.isin(drug_set)]
    logger.debug(
        'There are %i drugs and %i cell lines, with %i response records after trimming.'
         % (len(drug_set), len(cell_set), len(self.__drug_resp_df)))
    return
