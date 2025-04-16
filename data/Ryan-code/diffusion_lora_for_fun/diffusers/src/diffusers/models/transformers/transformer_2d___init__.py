@register_to_config
def __init__(self, num_attention_heads: int=16, attention_head_dim: int=88,
    in_channels: Optional[int]=None, out_channels: Optional[int]=None,
    num_layers: int=1, dropout: float=0.0, norm_num_groups: int=32,
    cross_attention_dim: Optional[int]=None, attention_bias: bool=False,
    sample_size: Optional[int]=None, num_vector_embeds: Optional[int]=None,
    patch_size: Optional[int]=None, activation_fn: str='geglu',
    num_embeds_ada_norm: Optional[int]=None, use_linear_projection: bool=
    False, only_cross_attention: bool=False, double_self_attention: bool=
    False, upcast_attention: bool=False, norm_type: str='layer_norm',
    norm_elementwise_affine: bool=True, norm_eps: float=1e-05,
    attention_type: str='default', caption_channels: int=None,
    interpolation_scale: float=None, use_additional_conditions: Optional[
    bool]=None):
    super().__init__()
    if patch_size is not None:
        if norm_type not in ['ada_norm', 'ada_norm_zero', 'ada_norm_single']:
            raise NotImplementedError(
                f"Forward pass is not implemented when `patch_size` is not None and `norm_type` is '{norm_type}'."
                )
        elif norm_type in ['ada_norm', 'ada_norm_zero'
            ] and num_embeds_ada_norm is None:
            raise ValueError(
                f'When using a `patch_size` and this `norm_type` ({norm_type}), `num_embeds_ada_norm` cannot be None.'
                )
    self.use_linear_projection = use_linear_projection
    self.interpolation_scale = interpolation_scale
    self.caption_channels = caption_channels
    self.num_attention_heads = num_attention_heads
    self.attention_head_dim = attention_head_dim
    self.inner_dim = (self.config.num_attention_heads * self.config.
        attention_head_dim)
    self.in_channels = in_channels
    self.out_channels = in_channels if out_channels is None else out_channels
    self.gradient_checkpointing = False
    if use_additional_conditions is None:
        if norm_type == 'ada_norm_single' and sample_size == 128:
            use_additional_conditions = True
        else:
            use_additional_conditions = False
    self.use_additional_conditions = use_additional_conditions
    self.is_input_continuous = in_channels is not None and patch_size is None
    self.is_input_vectorized = num_vector_embeds is not None
    self.is_input_patches = in_channels is not None and patch_size is not None
    if norm_type == 'layer_norm' and num_embeds_ada_norm is not None:
        deprecation_message = (
            f"The configuration file of this model: {self.__class__} is outdated. `norm_type` is either not set or incorrectly set to `'layer_norm'`. Make sure to set `norm_type` to `'ada_norm'` in the config. Please make sure to update the config accordingly as leaving `norm_type` might led to incorrect results in future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for the `transformer/config.json` file"
            )
        deprecate('norm_type!=num_embeds_ada_norm', '1.0.0',
            deprecation_message, standard_warn=False)
        norm_type = 'ada_norm'
    if self.is_input_continuous and self.is_input_vectorized:
        raise ValueError(
            f'Cannot define both `in_channels`: {in_channels} and `num_vector_embeds`: {num_vector_embeds}. Make sure that either `in_channels` or `num_vector_embeds` is None.'
            )
    elif self.is_input_vectorized and self.is_input_patches:
        raise ValueError(
            f'Cannot define both `num_vector_embeds`: {num_vector_embeds} and `patch_size`: {patch_size}. Make sure that either `num_vector_embeds` or `num_patches` is None.'
            )
    elif not self.is_input_continuous and not self.is_input_vectorized and not self.is_input_patches:
        raise ValueError(
            f'Has to define `in_channels`: {in_channels}, `num_vector_embeds`: {num_vector_embeds}, or patch_size: {patch_size}. Make sure that `in_channels`, `num_vector_embeds` or `num_patches` is not None.'
            )
    if self.is_input_continuous:
        self._init_continuous_input(norm_type=norm_type)
    elif self.is_input_vectorized:
        self._init_vectorized_inputs(norm_type=norm_type)
    elif self.is_input_patches:
        self._init_patched_inputs(norm_type=norm_type)
