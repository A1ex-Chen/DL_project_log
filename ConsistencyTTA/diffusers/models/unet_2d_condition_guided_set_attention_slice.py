def set_attention_slice(self, slice_size):
    """
        Enable sliced attention computation.

        When this option is enabled, the attention module will split the input tensor in slices, to compute
        attention in several steps. This is useful to save some memory in exchange for a small speed decrease.

        Args:
            slice_size (`str` or `int` or `list(int)`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two
                steps. "max"`, maximum amount of memory will be saved by running only one slice at a time.
                If a number is provided, uses as many slices as `num_attention_heads // slice_size`.
                In this case, `num_attention_heads` must be a multiple of `slice_size`.
        """
    sliceable_head_dims = []

    def fn_recursive_retrieve_sliceable_dims(module: torch.nn.Module):
        if hasattr(module, 'set_attention_slice'):
            sliceable_head_dims.append(module.sliceable_head_dim)
        for child in module.children():
            fn_recursive_retrieve_sliceable_dims(child)
    for module in self.children():
        fn_recursive_retrieve_sliceable_dims(module)
    num_sliceable_layers = len(sliceable_head_dims)
    if slice_size == 'auto':
        slice_size = [(dim // 2) for dim in sliceable_head_dims]
    elif slice_size == 'max':
        slice_size = num_sliceable_layers * [1]
    slice_size = num_sliceable_layers * [slice_size] if not isinstance(
        slice_size, list) else slice_size
    if len(slice_size) != len(sliceable_head_dims):
        raise ValueError(
            f'You have provided {len(slice_size)}, but {self.config} has {len(sliceable_head_dims)} different attention layers. Make sure to match `len(slice_size)` to be {len(sliceable_head_dims)}.'
            )
    for i in range(len(slice_size)):
        size = slice_size[i]
        dim = sliceable_head_dims[i]
        if size is not None and size > dim:
            raise ValueError(
                f'size {size} has to be smaller or equal to {dim}.')

    def fn_recursive_set_attention_slice(module: torch.nn.Module,
        slice_size: List[int]):
        if hasattr(module, 'set_attention_slice'):
            module.set_attention_slice(slice_size.pop())
        for child in module.children():
            fn_recursive_set_attention_slice(child, slice_size)
    reversed_slice_size = list(reversed(slice_size))
    for module in self.children():
        fn_recursive_set_attention_slice(module, reversed_slice_size)
