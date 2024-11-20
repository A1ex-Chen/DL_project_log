def _check_config(self, down_block_types: Tuple[str], up_block_types: Tuple
    [str], only_cross_attention: Union[bool, Tuple[bool]],
    block_out_channels: Tuple[int], layers_per_block: Union[int, Tuple[int]
    ], cross_attention_dim: Union[int, Tuple[int]],
    transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple[int]]],
    reverse_transformer_layers_per_block: bool, attention_head_dim: int,
    num_attention_heads: Optional[Union[int, Tuple[int]]]):
    if len(down_block_types) != len(up_block_types):
        raise ValueError(
            f'Must provide the same number of `down_block_types` as `up_block_types`. `down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}.'
            )
    if len(block_out_channels) != len(down_block_types):
        raise ValueError(
            f'Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}.'
            )
    if not isinstance(only_cross_attention, bool) and len(only_cross_attention
        ) != len(down_block_types):
        raise ValueError(
            f'Must provide the same number of `only_cross_attention` as `down_block_types`. `only_cross_attention`: {only_cross_attention}. `down_block_types`: {down_block_types}.'
            )
    if not isinstance(num_attention_heads, int) and len(num_attention_heads
        ) != len(down_block_types):
        raise ValueError(
            f'Must provide the same number of `num_attention_heads` as `down_block_types`. `num_attention_heads`: {num_attention_heads}. `down_block_types`: {down_block_types}.'
            )
    if not isinstance(attention_head_dim, int) and len(attention_head_dim
        ) != len(down_block_types):
        raise ValueError(
            f'Must provide the same number of `attention_head_dim` as `down_block_types`. `attention_head_dim`: {attention_head_dim}. `down_block_types`: {down_block_types}.'
            )
    if isinstance(cross_attention_dim, list) and len(cross_attention_dim
        ) != len(down_block_types):
        raise ValueError(
            f'Must provide the same number of `cross_attention_dim` as `down_block_types`. `cross_attention_dim`: {cross_attention_dim}. `down_block_types`: {down_block_types}.'
            )
    if not isinstance(layers_per_block, int) and len(layers_per_block) != len(
        down_block_types):
        raise ValueError(
            f'Must provide the same number of `layers_per_block` as `down_block_types`. `layers_per_block`: {layers_per_block}. `down_block_types`: {down_block_types}.'
            )
    if isinstance(transformer_layers_per_block, list
        ) and reverse_transformer_layers_per_block is None:
        for layer_number_per_block in transformer_layers_per_block:
            if isinstance(layer_number_per_block, list):
                raise ValueError(
                    "Must provide 'reverse_transformer_layers_per_block` if using asymmetrical UNet."
                    )
