def __post_init__(self):
    if (self.dataset_name is None and self.train_file is None and self.
        validation_file is None):
        raise ValueError(
            'Need either a dataset name or a training/validation file.')
    else:
        if self.train_file is not None:
            extension = self.train_file.split('.')[-1]
            if extension not in ['csv', 'json', 'txt']:
                raise ValueError(
                    '`train_file` should be a csv, a json or a txt file.')
        if self.validation_file is not None:
            extension = self.validation_file.split('.')[-1]
            if extension not in ['csv', 'json', 'txt']:
                raise ValueError(
                    '`validation_file` should be a csv, a json or a txt file.')
