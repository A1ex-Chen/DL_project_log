def test_spatial_transformer_default_ff_layers(self):
    spatial_transformer_block = Transformer2DModel(num_attention_heads=1,
        attention_head_dim=32, in_channels=32)
    assert spatial_transformer_block.transformer_blocks[0].ff.net[0
        ].__class__ == GEGLU
    assert spatial_transformer_block.transformer_blocks[0].ff.net[1
        ].__class__ == nn.Dropout
    assert spatial_transformer_block.transformer_blocks[0].ff.net[2
        ].__class__ == nn.Linear
    dim = 32
    inner_dim = 128
    assert spatial_transformer_block.transformer_blocks[0].ff.net[0
        ].proj.in_features == dim
    assert spatial_transformer_block.transformer_blocks[0].ff.net[0
        ].proj.out_features == inner_dim * 2
    assert spatial_transformer_block.transformer_blocks[0].ff.net[2
        ].in_features == inner_dim
    assert spatial_transformer_block.transformer_blocks[0].ff.net[2
        ].out_features == dim
