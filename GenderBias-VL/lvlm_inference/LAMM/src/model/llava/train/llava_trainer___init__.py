def __init__(self, batch_size: int, world_size: int, lengths: Optional[List
    [int]]=None, generator=None, group_by_modality: bool=False):
    if lengths is None:
        raise ValueError('Lengths must be provided.')
    self.batch_size = batch_size
    self.world_size = world_size
    self.lengths = lengths
    self.generator = generator
    self.group_by_modality = group_by_modality
