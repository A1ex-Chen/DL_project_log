def from_file(args, model):
    df_data = pd.read_csv(args.uq_infer_file, sep='\t')
    logger.info('data shape: {}'.format(df_data.shape))
    logger.info('Size of data to infer: {}'.format(df_data.shape))
    test_indices = range(df_data.shape[0])
    target_str = args.agg_dose or 'Growth'
    num_features_list = []
    feature_names_list = []
    for layer in model.layers:
        dict = layer.get_config()
        name = dict['name']
        if name.find('input') > -1:
            feature_names_list.append(name.split('.')[-1])
            size_ = dict['batch_input_shape']
            num_features_list.append(size_[1])
    feature_names_list.append('dragon7')
    test_gen = FromFileDataGenerator(df_data, test_indices, target_str,
        feature_names_list, num_features_list, batch_size=args.batch_size,
        shuffle=False)
    return test_gen
