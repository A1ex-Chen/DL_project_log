def __split_drug_resp(self):
    """self.__split_drug_resp()

        This function split training and validation drug response data based
        on the splitting specifications (disjoint drugs and/or disjoint cells).

        Upon the call, the function summarize all the drugs and cells. If
        disjoint (drugs/cells) is set to True, then it will split the list
        (of drugs/cells) into training/validation (drugs/cells).

        Otherwise, if disjoint (drugs/cells) is set to False, we make sure
        that the training/validation set contains the same (drugs/cells).

        Then it trims all three dataframes to make sure all the data in RAM is
        relevant for training/validation

        Note that the validation size is not guaranteed during splitting.
        What the function really splits by the ratio is the list of
        drugs/cell lines. Also, if both drugs and cell lines are marked
        disjoint, the function will split drug and cell lists with ratio of
        (validation_size ** 0.7).

        Warnings will be raise if the validation ratio is off too much.

        Returns:
            None
        """
    self.__trim_dataframes()
    cell_list = self.__drug_resp_df['CELLNAME'].unique().tolist()
    drug_list = self.__drug_resp_df['DRUG_ID'].unique().tolist()
    drug_anlys_dict = {idx: row.values for idx, row in get_drug_anlys_df(
        self.__data_root).iterrows()}
    drug_anlys_array = np.array([drug_anlys_dict[d] for d in drug_list])
    cell_type_dict = {idx: row.values for idx, row in get_cl_meta_df(self.
        __data_root)[['type']].iterrows()}
    cell_type_list = [cell_type_dict[c] for c in cell_list]
    if self.__disjoint_cells and self.__disjoint_drugs:
        adjusted_val_ratio = self.__validation_ratio ** 0.7
    else:
        adjusted_val_ratio = self.__validation_ratio
    split_kwargs = {'test_size': adjusted_val_ratio, 'random_state': self.
        __rand_state, 'shuffle': True}
    try:
        training_cell_list, validation_cell_list = train_test_split(cell_list,
            **split_kwargs, stratify=cell_type_list)
    except ValueError:
        logger.warning(
            'Failed to split %s cells in stratified way. Splitting randomly ...'
             % self.data_source)
        training_cell_list, validation_cell_list = train_test_split(cell_list,
            **split_kwargs)
    try:
        training_drug_list, validation_drug_list = train_test_split(drug_list,
            **split_kwargs, stratify=drug_anlys_array)
    except ValueError:
        logger.warning(
            'Failed to split %s drugs stratified on growth and correlation. Splitting solely on avg growth ...'
             % self.data_source)
        try:
            training_drug_list, validation_drug_list = train_test_split(
                drug_list, **split_kwargs, stratify=drug_anlys_array[:, 0])
        except ValueError:
            logger.warning(
                'Failed to split %s drugs on avg growth. Splitting solely on avg correlation ...'
                 % self.data_source)
            try:
                training_drug_list, validation_drug_list = train_test_split(
                    drug_list, **split_kwargs, stratify=drug_anlys_array[:, 1])
            except ValueError:
                logger.warning(
                    'Failed to split %s drugs on avg correlation. Splitting randomly ...'
                     % self.data_source)
                training_drug_list, validation_drug_list = train_test_split(
                    drug_list, **split_kwargs)
    if self.__disjoint_cells and self.__disjoint_drugs:
        training_drug_resp_df = self.__drug_resp_df.loc[self.__drug_resp_df
            ['CELLNAME'].isin(training_cell_list) & self.__drug_resp_df[
            'DRUG_ID'].isin(training_drug_list)]
        validation_drug_resp_df = self.__drug_resp_df.loc[self.
            __drug_resp_df['CELLNAME'].isin(validation_cell_list) & self.
            __drug_resp_df['DRUG_ID'].isin(validation_drug_list)]
    elif self.__disjoint_cells and not self.__disjoint_drugs:
        training_drug_resp_df = self.__drug_resp_df.loc[self.__drug_resp_df
            ['CELLNAME'].isin(training_cell_list)]
        validation_drug_resp_df = self.__drug_resp_df.loc[self.
            __drug_resp_df['CELLNAME'].isin(validation_cell_list)]
    elif not self.__disjoint_cells and self.__disjoint_drugs:
        training_drug_resp_df = self.__drug_resp_df.loc[self.__drug_resp_df
            ['DRUG_ID'].isin(training_drug_list)]
        validation_drug_resp_df = self.__drug_resp_df.loc[self.
            __drug_resp_df['DRUG_ID'].isin(validation_drug_list)]
    else:
        training_drug_resp_df, validation_drug_resp_df = train_test_split(self
            .__drug_resp_df, test_size=self.__validation_ratio,
            random_state=self.__rand_state, shuffle=False)
    if not self.__disjoint_cells:
        common_cells = set(training_drug_resp_df['CELLNAME'].unique()) & set(
            validation_drug_resp_df['CELLNAME'].unique())
        training_drug_resp_df = training_drug_resp_df.loc[training_drug_resp_df
            ['CELLNAME'].isin(common_cells)]
        validation_drug_resp_df = validation_drug_resp_df.loc[
            validation_drug_resp_df['CELLNAME'].isin(common_cells)]
    if not self.__disjoint_drugs:
        common_drugs = set(training_drug_resp_df['DRUG_ID'].unique()) & set(
            validation_drug_resp_df['DRUG_ID'].unique())
        training_drug_resp_df = training_drug_resp_df.loc[training_drug_resp_df
            ['DRUG_ID'].isin(common_drugs)]
        validation_drug_resp_df = validation_drug_resp_df.loc[
            validation_drug_resp_df['DRUG_ID'].isin(common_drugs)]
    validation_ratio = len(validation_drug_resp_df) / (len(
        training_drug_resp_df) + len(validation_drug_resp_df))
    if (validation_ratio < self.__validation_ratio * 0.8 or 
        validation_ratio > self.__validation_ratio * 1.2):
        logger.warning('Bad validation ratio: %.3f' % validation_ratio)
    self.__drug_resp_df = (training_drug_resp_df if self.training else
        validation_drug_resp_df)
