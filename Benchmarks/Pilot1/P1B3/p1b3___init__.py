def __init__(self, data, partition='train', batch_size=32, shape=None,
    concat=True, name='', cell_noise_sigma=None):
    """Initialize data

        Parameters
        ----------
        data: DataLoader object
            loaded data object containing original data frames for molecular, drug and response data
        partition: 'train', 'val', or 'test'
            partition of data to generate for
        batch_size: integer (default 32)
            batch size of generated data
        shape: None, '1d' or 'add_1d' (default None)
            keep original feature shapes, make them flat or add one extra dimension (for convolution or locally connected layers in some frameworks)
        concat: True or False (default True)
            concatenate all features if set to True
        cell_noise_sigma: float
            standard deviation of guassian noise to add to cell line features during training
        """
    self.lock = threading.Lock()
    self.data = data
    self.partition = partition
    self.batch_size = batch_size
    self.shape = shape
    self.concat = concat
    self.name = name
    self.cell_noise_sigma = cell_noise_sigma
    if partition == 'train':
        self.cycle = cycle(range(data.n_train))
        self.num_data = data.n_train
    elif partition == 'val':
        self.cycle = cycle(range(data.total)[-data.n_val:])
        self.num_data = data.n_val
    elif partition == 'test':
        self.cycle = cycle(range(data.total, data.total + data.n_test))
        self.num_data = data.n_test
    else:
        raise Exception('Data partition "{}" not recognized.'.format(partition)
            )
