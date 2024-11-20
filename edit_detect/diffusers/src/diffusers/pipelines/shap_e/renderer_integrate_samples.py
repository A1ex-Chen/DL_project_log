def integrate_samples(volume_range, ts, density, channels):
    """
    Function integrating the model output.

    Args:
        volume_range: Specifies the integral range [t0, t1]
        ts: timesteps
        density: torch.Tensor [batch_size, *shape, n_samples, 1]
        channels: torch.Tensor [batch_size, *shape, n_samples, n_channels]
    returns:
        channels: integrated rgb output weights: torch.Tensor [batch_size, *shape, n_samples, 1] (density
        *transmittance)[i] weight for each rgb output at [..., i, :]. transmittance: transmittance of this volume
    )
    """
    _, _, dt = volume_range.partition(ts)
    ddensity = density * dt
    mass = torch.cumsum(ddensity, dim=-2)
    transmittance = torch.exp(-mass[..., -1, :])
    alphas = 1.0 - torch.exp(-ddensity)
    Ts = torch.exp(torch.cat([torch.zeros_like(mass[..., :1, :]), -mass[...,
        :-1, :]], dim=-2))
    weights = alphas * Ts
    channels = torch.sum(channels * weights, dim=-2)
    return channels, weights, transmittance
