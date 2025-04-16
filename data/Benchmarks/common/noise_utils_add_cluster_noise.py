def add_cluster_noise(x_data, loc=0.0, scale=0.5, col_ids=[0], noise_type=
    'gaussian', row_ids=[0], y_noise_level=0.0):
    num_samples = len(row_ids)
    flip_count = 0
    for row_id in row_ids:
        if random.random() < y_noise_level:
            flip_count += 1
            for col_id in col_ids:
                print('added', noise_type, 'noise to row, column ', row_id,
                    col_id)
                print(x_data[row_id, col_id])
                if noise_type == 'gaussian':
                    train_noise = np.random.normal(loc, scale)
                elif noise_type == 'uniform':
                    train_noise = np.random.uniform(-1.0 * scale, scale)
                print(train_noise)
                x_data[row_id, col_id] = 1.0 * x_data[row_id, col_id
                    ] + 1.0 * train_noise
                print(x_data[row_id, col_id])
    y_noise_generated = float(flip_count) / float(num_samples)
    print(
        'Noise added to {} samples out of {}: {:06.4f} ({:06.4f} requested)\n'
        .format(flip_count, num_samples, y_noise_generated, y_noise_level))
    return x_data
