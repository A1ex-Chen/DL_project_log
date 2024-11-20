def add_column_noise(x_data, loc=0.0, scale=0.5, col_ids=[0], noise_type=
    'gaussian'):
    for col_id in col_ids:
        print('added', noise_type, 'noise to column ', col_id)
        print(x_data[:, col_id].T)
        if noise_type == 'gaussian':
            train_noise = np.random.normal(loc, scale, size=x_data.shape[0])
        elif noise_type == 'uniform':
            train_noise = np.random.uniform(-1.0 * scale, scale, size=
                x_data.shape[0])
        print(train_noise)
        x_data[:, col_id] = 1.0 * x_data[:, col_id] + 1.0 * train_noise.T
        print(x_data[:, col_id].T)
    return x_data
