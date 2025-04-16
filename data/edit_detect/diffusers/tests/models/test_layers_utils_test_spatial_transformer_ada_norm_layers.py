def test_spatial_transformer_ada_norm_layers(self):
    spatial_transformer_block = Transformer2DModel(num_attention_heads=1,
        attention_head_dim=32, in_channels=32, num_embeds_ada_norm=5)
    assert spatial_transformer_block.transformer_blocks[0
        ].norm1.__class__ == AdaLayerNorm
    assert spatial_transformer_block.transformer_blocks[0
        ].norm3.__class__ == nn.LayerNorm
