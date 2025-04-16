def test_bitfeedforward_forward_pass():
    bitffn = BitFeedForward(dim=512, ff_mult=4)
    x = torch.randn(1, 512)
    out = bitffn(x)
    assert out.shape == x.shape
