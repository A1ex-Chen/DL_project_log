def load_data(dataset):
    if dataset == 'mnist':
        (train_data, train_labels), (test_data, test_labels) = tfds.as_numpy(
            tfds.load('mnist', split=['train', 'test'], batch_size=-1,
            as_supervised=True))
    elif dataset == 'fashion_mnist':
        (train_data, train_labels), (test_data, test_labels) = tfds.as_numpy(
            tfds.load('fashion_mnist', split=['train', 'test'], batch_size=
            -1, as_supervised=True))
    elif dataset == 'cifar10':
        (train_data, train_labels), (test_data, test_labels) = tfds.as_numpy(
            tfds.load('cifar10', split=['train', 'test'], batch_size=-1,
            as_supervised=True))
    elif dataset == 'stl':
        (train_data, train_labels), (test_data, test_labels
            ), info = tfds.as_numpy(tfds.load('stl10', split=['train',
            'test'], batch_size=-1, as_supervised=True))
    elif dataset == 'celeb_a':
        (train_data, test_data), info = tfds.as_numpy(tfds.load('celeb_a',
            split=['train', 'test'], batch_size=-1, as_supervised=True))
        train_labels = None
        test_labels = None
    train_data = np.array(train_data, dtype=np.float32) / 255
    test_data = np.array(test_data, dtype=np.float32) / 255
    assert train_data.min() == 0.0
    assert train_data.max() == 1.0
    assert test_data.min() == 0.0
    assert test_data.max() == 1.0
    return train_data, train_labels, test_data, test_labels
