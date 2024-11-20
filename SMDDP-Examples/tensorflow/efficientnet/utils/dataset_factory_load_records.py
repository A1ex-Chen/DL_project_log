def load_records(self) ->tf.data.Dataset:
    """Return a dataset loading files with TFRecords."""
    if self._data_dir is None:
        raise ValueError('Dataset must specify a path for the data files.')
    file_pattern = os.path.join(self._data_dir, '{}*'.format(self._split))
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=False)
    return dataset
