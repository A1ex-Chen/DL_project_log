@property
def output_shape(self):
    if self.block_type == 'down':
        return 4, 32, 16, 16
    elif self.block_type == 'mid':
        return 4, 32, 32, 32
    elif self.block_type == 'up':
        return 4, 32, 64, 64
    raise ValueError(
        f"'{self.block_type}' is not a supported block_type. Set it to 'up', 'mid', or 'down'."
        )
