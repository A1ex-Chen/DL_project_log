def enable_free_init(self, num_iters: int=3, use_fast_sampling: bool=False,
    method: str='butterworth', order: int=4, spatial_stop_frequency: float=
    0.25, temporal_stop_frequency: float=0.25):
    """Enables the FreeInit mechanism as in https://arxiv.org/abs/2312.07537.

        This implementation has been adapted from the [official repository](https://github.com/TianxingWu/FreeInit).

        Args:
            num_iters (`int`, *optional*, defaults to `3`):
                Number of FreeInit noise re-initialization iterations.
            use_fast_sampling (`bool`, *optional*, defaults to `False`):
                Whether or not to speedup sampling procedure at the cost of probably lower quality results. Enables the
                "Coarse-to-Fine Sampling" strategy, as mentioned in the paper, if set to `True`.
            method (`str`, *optional*, defaults to `butterworth`):
                Must be one of `butterworth`, `ideal` or `gaussian` to use as the filtering method for the FreeInit low
                pass filter.
            order (`int`, *optional*, defaults to `4`):
                Order of the filter used in `butterworth` method. Larger values lead to `ideal` method behaviour
                whereas lower values lead to `gaussian` method behaviour.
            spatial_stop_frequency (`float`, *optional*, defaults to `0.25`):
                Normalized stop frequency for spatial dimensions. Must be between 0 to 1. Referred to as `d_s` in the
                original implementation.
            temporal_stop_frequency (`float`, *optional*, defaults to `0.25`):
                Normalized stop frequency for temporal dimensions. Must be between 0 to 1. Referred to as `d_t` in the
                original implementation.
        """
    self._free_init_num_iters = num_iters
    self._free_init_use_fast_sampling = use_fast_sampling
    self._free_init_method = method
    self._free_init_order = order
    self._free_init_spatial_stop_frequency = spatial_stop_frequency
    self._free_init_temporal_stop_frequency = temporal_stop_frequency
