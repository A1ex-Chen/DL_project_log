def __init__(self, num_channels, num_groups, eps=1e-05, affine=True):
    super().__init__(num_groups=num_groups, num_channels=num_channels, eps=
        eps, affine=affine)
