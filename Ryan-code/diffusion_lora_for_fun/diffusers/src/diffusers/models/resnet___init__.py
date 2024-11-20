def __init__(self, alpha: float, merge_strategy: str='learned_with_images',
    switch_spatial_to_temporal_mix: bool=False):
    super().__init__()
    self.merge_strategy = merge_strategy
    self.switch_spatial_to_temporal_mix = switch_spatial_to_temporal_mix
    if merge_strategy not in self.strategies:
        raise ValueError(f'merge_strategy needs to be in {self.strategies}')
    if self.merge_strategy == 'fixed':
        self.register_buffer('mix_factor', torch.Tensor([alpha]))
    elif self.merge_strategy == 'learned' or self.merge_strategy == 'learned_with_images':
        self.register_parameter('mix_factor', torch.nn.Parameter(torch.
            Tensor([alpha])))
    else:
        raise ValueError(f'Unknown merge strategy {self.merge_strategy}')
