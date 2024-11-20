def label_flip_correlated(y_data_categorical, y_noise_level, x_data,
    col_ids, threshold):
    for col_id in col_ids:
        flip_count = 0
        for i in range(0, y_data_categorical.shape[0]):
            if x_data[i][col_id] > threshold:
                if random.random() < y_noise_level:
                    print(i, y_data_categorical[i][:])
                    flip_count += 1
                    for j in range(y_data_categorical.shape[1]):
                        y_data_categorical[i][j] = int(not
                            y_data_categorical[i][j])
                    print(i, y_data_categorical[i][:])
        y_noise_generated = float(flip_count) / float(y_data_categorical.
            shape[0])
        print('Correlated label noise generation for feature {:d}:\n'.
            format(col_id))
        print(
            'Labels flipped on {} samples out of {}: {:06.4f} ({:06.4f} requested)\n'
            .format(flip_count, y_data_categorical.shape[0],
            y_noise_generated, y_noise_level))
    return y_data_categorical, y_noise_generated
