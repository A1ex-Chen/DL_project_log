def test_bitfeedforward_initialization():
    bitffn = BitFeedForward(dim=512, ff_mult=4)
    assert isinstance(bitffn.layer, nn.Sequential)
    assert len(bitffn.layer) == 3
    assert isinstance(bitffn.layer[0], BitLinear)
    assert isinstance(bitffn.layer[1], nn.GELU)
    assert isinstance(bitffn.layer[2], BitLinear)
    assert bitffn.layer[0].out_features == 512 * 4
    assert bitffn.layer[2].in_features == 512 * 4
