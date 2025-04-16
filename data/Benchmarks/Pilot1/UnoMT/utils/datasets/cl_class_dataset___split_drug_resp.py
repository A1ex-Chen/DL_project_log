def __split_drug_resp(self):
    """self.__split_drug_resp()

        Split training and validation dataframe for cell lines, stratified
        on tumor type. Note that after the split, our dataframe will only
        contain training/validation data based on training indicator.

        Returns:
            None
        """
    split_kwargs = {'test_size': self.__validation_ratio, 'random_state':
        self.__rand_state, 'shuffle': True}
    try:
        training_cl_df, validation_cl_df = train_test_split(self.__cl_df,
            **split_kwargs, stratify=self.__cl_df['type'].tolist())
    except ValueError:
        logger.warning(
            'Failed to split cell lines in stratified way. Splitting randomly ...'
            )
        training_cl_df, validation_cl_df = train_test_split(self.__cl_df,
            **split_kwargs)
    self.__cl_df = training_cl_df if self.training else validation_cl_df
