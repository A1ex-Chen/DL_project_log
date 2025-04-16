def __init__(self, data_dir: str='path/to/dir', batch_size: int=32,
    pin_memory: bool=False, num_workers: int=1, shuffle: bool=True,
    single_agent=False, smooth=True, split=96, max_num_agents=None):
    super().__init__()
    self.data_dir = Path(data_dir)
    self.check_data_dir()
    self.batch_size = batch_size
    self.pin_memory = pin_memory
    self.shuffle = shuffle
    self.num_workers = num_workers
    self.single_agent = single_agent
    self.split = split
    self.max_num_agents = max_num_agents
    transforms = [torch.Tensor]
    if smooth:
        transforms.append(smooth_sequence)
    if not single_agent:
        transforms.append(random_ordering)
    self.transform = Compose(transforms)
