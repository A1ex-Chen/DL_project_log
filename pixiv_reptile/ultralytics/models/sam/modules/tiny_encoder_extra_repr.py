def extra_repr(self) ->str:
    """Returns a string representation of the extra_repr function with the layer's parameters."""
    return (
        f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'
        )
