def test_spatial_transformer_attention_bias(self):
    spatial_transformer_block = Transformer2DModel(num_attention_heads=1,
        attention_head_dim=32, in_channels=32, attention_bias=True)
    assert spatial_transformer_block.transformer_blocks[0
        ].attn1.to_q.bias is not None
    assert spatial_transformer_block.transformer_blocks[0
        ].attn1.to_k.bias is not None
    assert spatial_transformer_block.transformer_blocks[0
        ].attn1.to_v.bias is not None
