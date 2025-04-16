def __repr__(self):
    return (
        f'{self._filename} with {len(self)} frames of size {self.frame_shape} at {self.frame_rate:1.2f} fps'
        )
