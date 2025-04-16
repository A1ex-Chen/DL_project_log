def __init__(self, num_input_features=4, use_norm=True, num_filters=[32, 
    128], with_distance=False, name='BinaryVoxel'):
    super(BinaryVoxel, self).__init__()
    self.name = name
