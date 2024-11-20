def label_flip(y_data_categorical, y_noise_level):
    flip_count = 0
    for i in range(0, y_data_categorical.shape[0]):
        if random.random() < y_noise_level:
            flip_count += 1
            for j in range(y_data_categorical.shape[1]):
                y_data_categorical[i][j] = int(not y_data_categorical[i][j])
    y_noise_generated = float(flip_count) / float(y_data_categorical.shape[0])
    print('Uncorrelated label noise generation:\n')
    print(
        'Labels flipped on {} samples out of {}: {:06.4f} ({:06.4f} requested)\n'
        .format(flip_count, y_data_categorical.shape[0], y_noise_generated,
        y_noise_level))
    return y_data_categorical, y_noise_generated
