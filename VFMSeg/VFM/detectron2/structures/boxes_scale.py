def scale(self, scale_x: float, scale_y: float) ->None:
    """
        Scale the box with horizontal and vertical scaling factors
        """
    self.tensor[:, 0::2] *= scale_x
    self.tensor[:, 1::2] *= scale_y
