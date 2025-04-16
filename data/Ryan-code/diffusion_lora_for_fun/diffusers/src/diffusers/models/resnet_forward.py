def forward(self, x_spatial: torch.Tensor, x_temporal: torch.Tensor,
    image_only_indicator: Optional[torch.Tensor]=None) ->torch.Tensor:
    alpha = self.get_alpha(image_only_indicator, x_spatial.ndim)
    alpha = alpha.to(x_spatial.dtype)
    if self.switch_spatial_to_temporal_mix:
        alpha = 1.0 - alpha
    x = alpha * x_spatial + (1.0 - alpha) * x_temporal
    return x
