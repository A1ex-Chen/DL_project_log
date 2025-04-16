def add_noise_new(data, labels, params):
    if params['noise_injection']:
        if params['label_noise']:
            if params['noise_correlated']:
                labels, y_noise_gen = label_flip_correlated(labels, params[
                    'label_noise'], data, params['feature_col'], params[
                    'feature_threshold'])
            else:
                labels, y_noise_gen = label_flip(labels, params['label_noise'])
        elif params['noise_gaussian']:
            data = add_gaussian_noise(data, 0, params['std_dev'])
        elif params['noise_cluster']:
            data = add_cluster_noise(data, loc=0.0, scale=params['std_dev'],
                col_ids=params['feature_col'], noise_type=params[
                'noise_type'], row_ids=params['sample_ids'], y_noise_level=
                params['label_noise'])
        elif params['noise_column']:
            data = add_column_noise(data, 0, params['std_dev'], col_ids=
                params['feature_col'], noise_type=params['noise_type'])
    return data, labels
