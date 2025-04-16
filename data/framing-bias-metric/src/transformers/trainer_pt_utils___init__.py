def __init__(self, world_size, num_samples, make_multiple_of=None,
    padding_index=-100):
    self.world_size = world_size
    self.num_samples = num_samples
    total_size = (world_size if make_multiple_of is None else world_size *
        make_multiple_of)
    self.total_samples = int(np.ceil(num_samples / total_size)) * total_size
    self.process_length = self.total_samples // world_size
    self._storage = None
    self._offsets = None
    self.padding_index = padding_index
