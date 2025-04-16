def __init__(self, d_model: int, n_heads: int, attn_impl: str='triton',
    clip_qkv: Optional[float]=None, qk_ln: bool=False, softmax_scale:
    Optional[float]=None, attn_pdrop: float=0.0, low_precision_layernorm:
    bool=False, verbose: int=0, device: Optional[str]=None):
    super().__init__()
    self.attn_impl = attn_impl
    self.clip_qkv = clip_qkv
    self.qk_ln = qk_ln
    self.d_model = d_model
    self.n_heads = n_heads
    self.head_dim = d_model // n_heads
    self.softmax_scale = softmax_scale
    if self.softmax_scale is None:
        self.softmax_scale = 1 / math.sqrt(self.head_dim)
    self.attn_dropout_p = attn_pdrop
    self.Wqkv = nn.Linear(d_model, d_model + 2 * self.head_dim, device=device)
    fuse_splits = d_model, d_model + self.head_dim
    self.Wqkv._fused = 0, fuse_splits
    if self.qk_ln:
        layernorm_class = (LPLayerNorm if low_precision_layernorm else nn.
            LayerNorm)
        self.q_ln = layernorm_class(d_model, device=device)
        self.k_ln = layernorm_class(self.head_dim, device=device)
    if self.attn_impl == 'flash':
        self.attn_fn = flash_attn_fn
    elif self.attn_impl == 'triton':
        self.attn_fn = triton_flash_attn_fn
        if verbose:
            warnings.warn(
                'While `attn_impl: triton` can be faster than `attn_impl: flash` '
                 +
                'it uses more memory. When training larger models this can trigger '
                 +
                'alloc retries which hurts performance. If encountered, we recommend '
                 +
                'using `attn_impl: flash` if your model does not use `alibi` or `prefix_lm`.'
                )
    elif self.attn_impl == 'torch':
        self.attn_fn = scaled_multihead_dot_product_attention
        if torch.cuda.is_available() and verbose:
            warnings.warn(
                'Using `attn_impl: torch`. If your model does not use `alibi` or '
                 +
                '`prefix_lm` we recommend using `attn_impl: flash` otherwise '
                 + 'we recommend using `attn_impl: triton`.')
    else:
        raise ValueError(f'attn_impl={attn_impl!r} is an invalid setting.')
    self.out_proj = nn.Linear(self.d_model, self.d_model, device=device)
    self.out_proj._is_residual = True
