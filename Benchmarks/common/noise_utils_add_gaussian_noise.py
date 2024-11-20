def add_gaussian_noise(x_data, loc=0.0, scale=0.5):
    print('added gaussian noise')
    train_noise = np.random.normal(loc, scale, size=x_data.shape)
    x_data = x_data + train_noise
    return x_data
