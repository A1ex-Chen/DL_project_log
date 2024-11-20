def build_feature_list(self, single=False):
    input_features = collections.OrderedDict()
    feature_shapes = collections.OrderedDict()
    if not self.agg_dose:
        doses = ['dose1', 'dose2'] if not single else ['dose1']
        for dose in doses:
            input_features[dose] = 'dose'
            feature_shapes['dose'] = 1,
    if self.encode_response_source:
        input_features['response.source'] = 'response.source'
        feature_shapes['response.source'] = self.df_source.shape[1] - 1,
    for fea in self.cell_features:
        feature_type = 'cell.' + fea
        feature_name = 'cell.' + fea
        df_cell = getattr(self, self.cell_df_dict[fea])
        input_features[feature_name] = feature_type
        feature_shapes[feature_type] = df_cell.shape[1] - 1,
    drugs = ['drug1', 'drug2'] if not single else ['drug1']
    for drug in drugs:
        for fea in self.drug_features:
            feature_type = 'drug.' + fea
            feature_name = drug + '.' + fea
            df_drug = getattr(self, self.drug_df_dict[fea])
            input_features[feature_name] = feature_type
            feature_shapes[feature_type] = df_drug.shape[1] - 1,
    input_dim = sum([np.prod(feature_shapes[x]) for x in input_features.
        values()])
    self.input_features = input_features
    self.feature_shapes = feature_shapes
    self.input_dim = input_dim
    logger.info('Input features shapes:')
    for k, v in self.input_features.items():
        logger.info('  {}: {}'.format(k, self.feature_shapes[v]))
    logger.info('Total input dimensions: {}'.format(self.input_dim))
