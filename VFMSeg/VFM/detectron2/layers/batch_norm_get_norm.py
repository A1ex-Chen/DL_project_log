def get_norm(norm, out_channels):
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module.

    Returns:
        nn.Module or None: the normalization layer
    """
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {'BN': BatchNorm2d, 'SyncBN': NaiveSyncBatchNorm if env.
            TORCH_VERSION <= (1, 5) else nn.SyncBatchNorm, 'FrozenBN':
            FrozenBatchNorm2d, 'GN': lambda channels: nn.GroupNorm(32,
            channels), 'nnSyncBN': nn.SyncBatchNorm, 'naiveSyncBN':
            NaiveSyncBatchNorm, 'naiveSyncBN_N': lambda channels:
            NaiveSyncBatchNorm(channels, stats_mode='N')}[norm]
    return norm(out_channels)
