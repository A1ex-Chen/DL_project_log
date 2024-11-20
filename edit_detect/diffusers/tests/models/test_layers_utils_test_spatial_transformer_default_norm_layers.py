def test_spatial_transformer_default_norm_layers(self):
    spatial_transformer_block = Transformer2DModel(num_attention_heads=1,
        attention_head_dim=32, in_channels=32)
    assert spatial_transformer_block.transformer_blocks[0
        ].norm1.__class__ == nn.LayerNorm
    assert spatial_transformer_block.transformer_blocks[0
        ].norm3.__class__ == nn.LayerNorm
