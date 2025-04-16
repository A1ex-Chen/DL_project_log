def load(self, cache=None, ncols=None, scaling='std', dropna=None, agg_dose
    =None, embed_feature_source=True, encode_response_source=True,
    cell_features=['rnaseq'], drug_features=['descriptors', 'fingerprints'],
    cell_feature_subset_path=None, drug_feature_subset_path=None,
    drug_lower_response=1, drug_upper_response=-1, drug_response_span=0,
    drug_median_response_min=-1, drug_median_response_max=1,
    use_landmark_genes=False, use_filtered_genes=False, use_exported_data=
    None, preprocess_rnaseq=None, single=False, train_sources=['GDSC',
    'CTRP', 'ALMANAC'], test_sources=['train'], partition_by='drug_pair'):
    params = locals().copy()
    del params['self']
    if not cell_features or 'none' in [x.lower() for x in cell_features]:
        cell_features = []
    if not drug_features or 'none' in [x.lower() for x in drug_features]:
        drug_features = []
    if cache and self.load_from_cache(cache, params):
        self.build_feature_list(single=single)
        return
    if use_exported_data is not None:
        with pd.HDFStore(use_exported_data, 'r') as store:
            if '/model' in store.keys():
                self.input_features = store.get_storer('model'
                    ).attrs.input_features
                self.feature_shapes = store.get_storer('model'
                    ).attrs.feature_shapes
                self.input_dim = sum([np.prod(self.feature_shapes[x]) for x in
                    self.input_features.values()])
                self.test_sep_sources = []
                return
            else:
                logger.warning(
                    """
Exported dataset does not have model info. Please rebuild the dataset.
"""
                    )
                raise ValueError('Could not load model info from the dataset:',
                    use_exported_data)
    logger.info('Loading data from scratch ...')
    if agg_dose:
        df_response = load_aggregated_single_response(target=agg_dose,
            combo_format=True)
    else:
        df_response = load_combined_dose_response()
    if logger.isEnabledFor(logging.INFO):
        logger.info('Summary of combined dose response by source:')
        logger.info(summarize_response_data(df_response, target=agg_dose))
    all_sources = df_response['Source'].unique()
    df_source = encode_sources(all_sources)
    if 'all' in train_sources:
        train_sources = all_sources
    if 'all' in test_sources:
        test_sources = all_sources
    elif 'train' in test_sources:
        test_sources = train_sources
    train_sep_sources = [x for x in all_sources for y in train_sources if x
        .startswith(y)]
    test_sep_sources = [x for x in all_sources for y in test_sources if x.
        startswith(y)]
    ids1 = df_response[['Drug1']].drop_duplicates().rename(columns={'Drug1':
        'Drug'})
    ids2 = df_response[['Drug2']].drop_duplicates().rename(columns={'Drug2':
        'Drug'})
    df_drugs_with_response = pd.concat([ids1, ids2]).drop_duplicates().dropna(
        ).reset_index(drop=True)
    df_cells_with_response = df_response[['Sample']].drop_duplicates(
        ).reset_index(drop=True)
    logger.info(
        'Combined raw dose response data has %d unique samples and %d unique drugs'
        , df_cells_with_response.shape[0], df_drugs_with_response.shape[0])
    if agg_dose:
        df_selected_drugs = None
    else:
        logger.info(
            'Limiting drugs to those with response min <= %g, max >= %g, span >= %g, median_min <= %g, median_max >= %g ...'
            , drug_lower_response, drug_upper_response, drug_response_span,
            drug_median_response_min, drug_median_response_max)
        df_selected_drugs = select_drugs_with_response_range(df_response,
            span=drug_response_span, lower=drug_lower_response, upper=
            drug_upper_response, lower_median=drug_median_response_min,
            upper_median=drug_median_response_max)
        logger.info('Selected %d drugs from %d', df_selected_drugs.shape[0],
            df_response['Drug1'].nunique())
    cell_feature_subset = read_set_from_file(cell_feature_subset_path)
    drug_feature_subset = read_set_from_file(drug_feature_subset_path)
    for fea in cell_features:
        fea = fea.lower()
        if fea == 'rnaseq' or fea == 'expression':
            df_cell_rnaseq = load_cell_rnaseq(ncols=ncols, scaling=scaling,
                use_landmark_genes=use_landmark_genes, use_filtered_genes=
                use_filtered_genes, feature_subset=cell_feature_subset,
                preprocess_rnaseq=preprocess_rnaseq, embed_feature_source=
                embed_feature_source)
    for fea in drug_features:
        fea = fea.lower()
        if fea == 'descriptors':
            df_drug_desc = load_drug_descriptors(ncols=ncols, scaling=
                scaling, dropna=dropna, feature_subset=drug_feature_subset)
        elif fea == 'fingerprints':
            df_drug_fp = load_drug_fingerprints(ncols=ncols, scaling=
                scaling, dropna=dropna, feature_subset=drug_feature_subset)
        elif fea == 'mordred':
            df_drug_mordred = load_mordred_descriptors(ncols=ncols, scaling
                =scaling, dropna=dropna, feature_subset=drug_feature_subset)
    cell_df_dict = {'rnaseq': 'df_cell_rnaseq'}
    drug_df_dict = {'descriptors': 'df_drug_desc', 'fingerprints':
        'df_drug_fp', 'mordred': 'df_drug_mordred'}
    logger.info('Filtering drug response data...')
    df_cell_ids = df_cells_with_response
    for fea in cell_features:
        df_cell = locals()[cell_df_dict[fea]]
        df_cell_ids = df_cell_ids.merge(df_cell[['Sample']]).drop_duplicates()
    logger.info('  %d molecular samples with feature and response data',
        df_cell_ids.shape[0])
    df_drug_ids = df_drugs_with_response
    for fea in drug_features:
        df_drug = locals()[drug_df_dict[fea]]
        df_drug_ids = df_drug_ids.merge(df_drug[['Drug']]).drop_duplicates()
    if df_selected_drugs is not None:
        df_drug_ids = df_drug_ids.merge(df_selected_drugs).drop_duplicates()
    logger.info('  %d selected drugs with feature and response data',
        df_drug_ids.shape[0])
    df_response = df_response[df_response['Sample'].isin(df_cell_ids[
        'Sample']) & df_response['Drug1'].isin(df_drug_ids['Drug']) & (
        df_response['Drug2'].isin(df_drug_ids['Drug']) | df_response[
        'Drug2'].isnull())]
    df_response = df_response[df_response['Source'].isin(train_sep_sources +
        test_sep_sources)]
    df_response.reset_index(drop=True, inplace=True)
    if logger.isEnabledFor(logging.INFO):
        logger.info('Summary of filtered dose response by source:')
        logger.info(summarize_response_data(df_response, target=agg_dose))
    df_response = df_response.assign(Group=assign_partition_groups(
        df_response, partition_by))
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
    for var in (list(drug_df_dict.values()) + list(cell_df_dict.values())):
        value = locals().get(var)
        if value is not None:
            setattr(self, var, value)
    self.build_feature_list(single=single)
    if cache:
        self.save_to_cache(cache, params)
