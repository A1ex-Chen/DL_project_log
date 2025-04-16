def load_data():
    path = 'npydata'
    x_train = np.load(os.path.join(path, 'imgs_train.npy'))
    y_train = np.load(os.path.join(path, 'imgs_mask_train.npy'))
    x_train = x_train.astype('float32')
    y_train = y_train.astype('float32')
    x_train /= 255
    y_train /= 255
    return x_train, y_train
