def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
    """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
    super().__init__()
    if d_model % n_heads != 0:
        raise ValueError(
            'd_model must be divisible by n_heads, but got {} and {}'.
            format(d_model, n_heads))
    _d_per_head = d_model // n_heads
    if not _is_power_of_2(_d_per_head):
        warnings.warn(
            "You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 which is more efficient in our CUDA implementation."
            )
    self.im2col_step = 128
    self.d_model = d_model
    self.n_levels = n_levels
    self.n_heads = n_heads
    self.n_points = n_points
    self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels *
        n_points * 2)
    self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
    self.value_proj = nn.Linear(d_model, d_model)
    self.output_proj = nn.Linear(d_model, d_model)
    self._reset_parameters()
