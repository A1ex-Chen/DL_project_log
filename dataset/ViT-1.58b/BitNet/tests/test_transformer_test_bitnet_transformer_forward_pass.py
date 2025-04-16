def test_bitnet_transformer_forward_pass():
    bitnet = BitNetTransformer(num_tokens=20000, dim=512, heads=8, depth=6,
        ff_mult=4)
    x = torch.randn(1, 100, 512)
    out = bitnet(x)
    assert out.shape == x.shape
