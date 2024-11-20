def down_sample(self, factor=2):
    if not self.resolution % factor == 0:
        raise ValueError('Resolution must be divisible by factor.')
    new_data = block_reduce(self.data, (factor,) * 3, np.max)
    return VoxelGrid(new_data, self.loc, self.scale)
