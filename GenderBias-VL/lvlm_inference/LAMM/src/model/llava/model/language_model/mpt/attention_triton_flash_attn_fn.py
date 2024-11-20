def triton_flash_attn_fn(query, key, value, n_heads, past_key_value=None,
    softmax_scale=None, attn_bias=None, key_padding_mask=None, is_causal=
    False, dropout_p=0.0, training=False, needs_weights=False, multiquery=False
    ):
    try:
        from .flash_attn_triton import flash_attn_func
    except:
        _installed = False
        if version.parse(torch.__version__) < version.parse('2.0.0'):
            _installed = True
            try:
                from flash_attn.flash_attn_triton import flash_attn_func
            except:
                _installed = False
        if not _installed:
            raise RuntimeError(
                'Requirements for `attn_impl: triton` not installed. Either (1) have a CUDA-compatible GPU and `pip install .[gpu]` if installing from llm-foundry source or `pip install triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir#subdirectory=python` if installing from pypi, or (2) use torch attn model.attn_config.attn_impl=torch (torch attn_impl will be slow). Note: (1) requires you have CMake and PyTorch already installed.'
                )
    check_valid_inputs(query, key, value)
    if past_key_value is not None:
        if len(past_key_value) != 0:
            key = torch.cat([past_key_value[0], key], dim=1)
            value = torch.cat([past_key_value[1], value], dim=1)
        past_key_value = key, value
    if attn_bias is not None:
        _s_q = max(0, attn_bias.size(2) - query.size(1))
        _s_k = max(0, attn_bias.size(3) - key.size(1))
        attn_bias = attn_bias[:, :, _s_q:, _s_k:]
    if dropout_p:
        raise NotImplementedError(
            f'Dropout not implemented for attn_impl: triton.')
    if needs_weights:
        raise NotImplementedError(
            f'attn_impl: triton cannot return attn weights.')
    if key_padding_mask is not None:
        warnings.warn(
            'Propagating key_padding_mask to the attention module ' +
            'and applying it within the attention module can cause ' +
            'unnecessary computation/memory usage. Consider integrating ' +
            'into attn_bias once and passing that to each attention ' +
            'module instead.')
        b_size, s_k = key_padding_mask.shape[:2]
        if attn_bias is None:
            attn_bias = query.new_zeros(b_size, 1, 1, s_k)
        attn_bias = attn_bias.masked_fill(~key_padding_mask.view((b_size, 1,
            1, s_k)), torch.finfo(query.dtype).min)
    query = rearrange(query, 'b s (h d) -> b s h d', h=n_heads)
    key = rearrange(key, 'b s (h d) -> b s h d', h=1 if multiquery else n_heads
        )
    value = rearrange(value, 'b s (h d) -> b s h d', h=1 if multiquery else
        n_heads)
    if multiquery:
        key = key.expand(*key.shape[:2], n_heads, key.size(-1))
        value = value.expand(*value.shape[:2], n_heads, value.size(-1))
    reset_is_causal = _reset_is_causal(query.size(1), key.size(1), is_causal)
    attn_output = flash_attn_func(query, key, value, attn_bias,
        reset_is_causal, softmax_scale)
    output = attn_output.view(*attn_output.shape[:2], -1)
    return output, None, past_key_value
