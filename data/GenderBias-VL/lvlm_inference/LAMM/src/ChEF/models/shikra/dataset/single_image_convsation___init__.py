def __init__(self, *args, dataset_generator: Type[Dataset], **kwargs):
    super().__init__(*args, **kwargs)
    self.dataset_generator = dataset_generator
    self.dataset = None
