def save_cache(cache_file, x_train, y_train, x_val, y_val, x_test, y_test,
    x_labels, y_labels):
    with h5py.File(cache_file, 'w') as hf:
        hf.create_dataset('x_train', data=x_train)
        hf.create_dataset('y_train', data=y_train)
        hf.create_dataset('x_val', data=x_val)
        hf.create_dataset('y_val', data=y_val)
        hf.create_dataset('x_test', data=x_test)
        hf.create_dataset('y_test', data=y_test)
        hf.create_dataset('x_labels', (len(x_labels), 1), 'S100', data=[x.
            encode('ascii', 'ignore') for x in x_labels])
        hf.create_dataset('y_labels', (len(y_labels), 1), 'S100', data=[x.
            encode('ascii', 'ignore') for x in y_labels])
