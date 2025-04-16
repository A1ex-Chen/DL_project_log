def initialize_if_needed(self):
    """
        lazy initialize for big in-memory python object due to python 'copy-on-read' behavior
        when num_worker > 0. refer: https://github.com/pytorch/pytorch/issues/13246
        """
    if self.dataset is None:
        self.dataset = self.dataset_generator()
