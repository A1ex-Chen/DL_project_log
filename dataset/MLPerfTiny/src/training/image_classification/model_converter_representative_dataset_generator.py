def representative_dataset_generator():
    (train_data, train_filenames, train_labels, test_data, test_filenames,
        test_labels, label_names) = train.load_cifar_10_data(cifar_10_dir)
    _idx = np.load('calibration_samples_idxs.npy')
    for i in _idx:
        sample_img = np.expand_dims(np.array(test_data[i], dtype=np.float32
            ), axis=0)
        yield [sample_img]
