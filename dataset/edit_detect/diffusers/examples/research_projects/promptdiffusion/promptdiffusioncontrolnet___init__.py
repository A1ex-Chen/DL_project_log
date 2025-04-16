@register_to_config
def __init__(self, in_channels: int=4, conditioning_channels: int=3,
    flip_sin_to_cos: bool=True, freq_shift: int=0, down_block_types: Tuple[
    str, ...]=('CrossAttnDownBlock2D', 'CrossAttnDownBlock2D',
    'CrossAttnDownBlock2D', 'DownBlock2D'), mid_block_type: Optional[str]=
    'UNetMidBlock2DCrossAttn', only_cross_attention: Union[bool, Tuple[bool
    ]]=False, block_out_channels: Tuple[int, ...]=(320, 640, 1280, 1280),
    layers_per_block: int=2, downsample_padding: int=1,
    mid_block_scale_factor: float=1, act_fn: str='silu', norm_num_groups:
    Optional[int]=32, norm_eps: float=1e-05, cross_attention_dim: int=1280,
    transformer_layers_per_block: Union[int, Tuple[int, ...]]=1,
    encoder_hid_dim: Optional[int]=None, encoder_hid_dim_type: Optional[str
    ]=None, attention_head_dim: Union[int, Tuple[int, ...]]=8,
    num_attention_heads: Optional[Union[int, Tuple[int, ...]]]=None,
    use_linear_projection: bool=False, class_embed_type: Optional[str]=None,
    addition_embed_type: Optional[str]=None, addition_time_embed_dim:
    Optional[int]=None, num_class_embeds: Optional[int]=None,
    upcast_attention: bool=False, resnet_time_scale_shift: str='default',
    projection_class_embeddings_input_dim: Optional[int]=None,
    controlnet_conditioning_channel_order: str='rgb',
    conditioning_embedding_out_channels: Optional[Tuple[int, ...]]=(16, 32,
    96, 256), global_pool_conditions: bool=False,
    addition_embed_type_num_heads: int=64):
    super().__init__(in_channels, conditioning_channels, flip_sin_to_cos,
        freq_shift, down_block_types, mid_block_type, only_cross_attention,
        block_out_channels, layers_per_block, downsample_padding,
        mid_block_scale_factor, act_fn, norm_num_groups, norm_eps,
        cross_attention_dim, transformer_layers_per_block, encoder_hid_dim,
        encoder_hid_dim_type, attention_head_dim, num_attention_heads,
        use_linear_projection, class_embed_type, addition_embed_type,
        addition_time_embed_dim, num_class_embeds, upcast_attention,
        resnet_time_scale_shift, projection_class_embeddings_input_dim,
        controlnet_conditioning_channel_order,
        conditioning_embedding_out_channels, global_pool_conditions,
        addition_embed_type_num_heads)
    self.controlnet_query_cond_embedding = ControlNetConditioningEmbedding(
        conditioning_embedding_channels=block_out_channels[0],
        block_out_channels=conditioning_embedding_out_channels,
        conditioning_channels=3)
