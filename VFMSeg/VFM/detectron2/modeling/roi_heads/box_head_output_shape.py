@property
@torch.jit.unused
def output_shape(self):
    """
        Returns:
            ShapeSpec: the output feature shape
        """
    o = self._output_size
    if isinstance(o, int):
        return ShapeSpec(channels=o)
    else:
        return ShapeSpec(channels=o[0], height=o[1], width=o[2])
