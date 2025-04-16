def scale(self, scale_x: float, scale_y: float) ->None:
    """
        Scale the rotated box with horizontal and vertical scaling factors
        Note: when scale_factor_x != scale_factor_y,
        the rotated box does not preserve the rectangular shape when the angle
        is not a multiple of 90 degrees under resize transformation.
        Instead, the shape is a parallelogram (that has skew)
        Here we make an approximation by fitting a rotated rectangle to the parallelogram.
        """
    self.tensor[:, 0] *= scale_x
    self.tensor[:, 1] *= scale_y
    theta = self.tensor[:, 4] * math.pi / 180.0
    c = torch.cos(theta)
    s = torch.sin(theta)
    self.tensor[:, 2] *= torch.sqrt((scale_x * c) ** 2 + (scale_y * s) ** 2)
    self.tensor[:, 3] *= torch.sqrt((scale_x * s) ** 2 + (scale_y * c) ** 2)
    self.tensor[:, 4] = torch.atan2(scale_x * s, scale_y * c) * 180 / math.pi
