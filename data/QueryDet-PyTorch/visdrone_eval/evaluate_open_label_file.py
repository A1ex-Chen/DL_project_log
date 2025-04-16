def open_label_file(path, dtype=np.float32):
    label = np.loadtxt(path, delimiter=',', dtype=dtype, ndmin=2, usecols=
        range(8))
    if not len(label):
        label = label.reshape(0, 8)
    return label
