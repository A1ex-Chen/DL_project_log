def __post_init__(self):
    if (self.dataset_name is None and self.train_file is None and self.
        validation_file is None and self.test_file is None):
        raise ValueError(
            'Need either a dataset name or a training/validation file/test_file.'
            )
    else:
        if self.train_file is not None:
            extension = self.train_file.split('.')[-1]
            assert extension in ['csv', 'json'
                ], '`train_file` should be a csv or a json file.'
        if self.validation_file is not None:
            extension = self.validation_file.split('.')[-1]
            assert extension in ['csv', 'json'
                ], '`validation_file` should be a csv or a json file.'
        if self.test_file is not None:
            extension = self.test_file.split('.')[-1]
            assert extension in ['csv', 'json'
                ], '`test_file` should be a csv or a json file.'
