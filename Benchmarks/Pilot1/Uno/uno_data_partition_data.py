def partition_data(self, partition_by=None, cv_folds=1, train_split=0.7,
    val_split=0.2, cell_types=None, by_cell=None, by_drug=None,
    cell_subset_path=None, drug_subset_path=None, exclude_cells=[],
    exclude_drugs=[], exclude_indices=[]):
    seed = self.seed
    train_sep_sources = self.train_sep_sources
    test_sep_sources = self.test_sep_sources
    df_response = self.df_response
    if not partition_by:
        if by_drug and by_cell:
            partition_by = 'index'
        elif by_drug:
            partition_by = 'cell'
        else:
            partition_by = 'drug_pair'
    if exclude_cells != []:
        df_response = df_response[~df_response['Sample'].isin(exclude_cells)]
    if exclude_drugs != []:
        if np.isin('Drug', df_response.columns.values):
            df_response = df_response[~df_response['Drug1'].isin(exclude_drugs)
                ]
        else:
            df_response = df_response[~df_response['Drug1'].isin(
                exclude_drugs) & ~df_response['Drug2'].isin(exclude_drugs)]
    if exclude_indices != []:
        df_response = df_response.drop(exclude_indices, axis=0)
        logger.info('Excluding indices specified')
    if partition_by != self.partition_by:
        df_response = df_response.assign(Group=assign_partition_groups(
            df_response, partition_by))
    mask = df_response['Source'].isin(train_sep_sources)
    test_mask = df_response['Source'].isin(test_sep_sources)
    if by_drug:
        drug_ids = drug_name_to_ids(by_drug)
        logger.info('Mapped drug IDs for %s: %s', by_drug, drug_ids)
        mask &= df_response['Drug1'].isin(drug_ids) & df_response['Drug2'
            ].isnull()
        test_mask &= df_response['Drug1'].isin(drug_ids) & df_response['Drug2'
            ].isnull()
    if by_cell:
        cell_ids = cell_name_to_ids(by_cell)
        logger.info('Mapped sample IDs for %s: %s', by_cell, cell_ids)
        mask &= df_response['Sample'].isin(cell_ids)
        test_mask &= df_response['Sample'].isin(cell_ids)
    if cell_subset_path:
        cell_subset = read_set_from_file(cell_subset_path)
        mask &= df_response['Sample'].isin(cell_subset)
        test_mask &= df_response['Sample'].isin(cell_subset)
    if drug_subset_path:
        drug_subset = read_set_from_file(drug_subset_path)
        mask &= df_response['Drug1'].isin(drug_subset) & (df_response[
            'Drug2'].isnull() | df_response['Drug2'].isin(drug_subset))
        test_mask &= df_response['Drug1'].isin(drug_subset) & (df_response[
            'Drug2'].isnull() | df_response['Drug2'].isin(drug_subset))
    if cell_types:
        df_type = load_cell_metadata()
        cell_ids = set()
        for cell_type in cell_types:
            cells = df_type[~df_type['TUMOR_TYPE'].isnull() & df_type[
                'TUMOR_TYPE'].str.contains(cell_type, case=False)]
            cell_ids |= set(cells['ANL_ID'].tolist())
            logger.info('Mapped sample tissue types for %s: %s', cell_type,
                set(cells['TUMOR_TYPE'].tolist()))
        mask &= df_response['Sample'].isin(cell_ids)
        test_mask &= df_response['Sample'].isin(cell_ids)
    df_group = df_response[mask]['Group'].drop_duplicates().reset_index(drop
        =True)
    if cv_folds > 1:
        selector = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    else:
        selector = ShuffleSplit(n_splits=1, train_size=train_split,
            test_size=val_split, random_state=seed)
    splits = selector.split(df_group)
    train_indexes = []
    val_indexes = []
    test_indexes = []
    for index, (train_group_index, val_group_index) in enumerate(splits):
        train_groups = set(df_group.values[train_group_index])
        val_groups = set(df_group.values[val_group_index])
        train_index = df_response.index[df_response['Group'].isin(
            train_groups) & mask]
        val_index = df_response.index[df_response['Group'].isin(val_groups) &
            mask]
        test_index = df_response.index[~df_response['Group'].isin(
            train_groups) & ~df_response['Group'].isin(val_groups) & test_mask]
        train_indexes.append(train_index)
        val_indexes.append(val_index)
        test_indexes.append(test_index)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                'CV fold %d: train data = %s, val data = %s, test data = %s',
                index, train_index.shape[0], val_index.shape[0], test_index
                .shape[0])
            logger.debug('  train groups (%d): %s', df_response.loc[
                train_index]['Group'].nunique(), df_response.loc[
                train_index]['Group'].unique())
            logger.debug('  val groups ({%d}): %s', df_response.loc[
                val_index]['Group'].nunique(), df_response.loc[val_index][
                'Group'].unique())
            logger.debug('  test groups ({%d}): %s', df_response.loc[
                test_index]['Group'].nunique(), df_response.loc[test_index]
                ['Group'].unique())
    self.partition_by = partition_by
    self.cv_folds = cv_folds
    self.train_indexes = train_indexes
    self.val_indexes = val_indexes
    self.test_indexes = test_indexes
