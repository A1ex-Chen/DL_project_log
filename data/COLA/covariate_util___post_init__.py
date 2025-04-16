def __post_init__(self):
    if self.dataset_name is None and self.data_file is None:
        raise ValueError(
            'Need either a dataset name or a training/validation file.')
    elif self.data_file is not None:
        extension = self.data_file.split('.')[-1]
        assert extension in ['csv', 'json'
            ], '`data_file` should be a csv or a json file.'
