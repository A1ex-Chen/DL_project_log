@pytest.mark.skipif(not TORCH_2_0, reason=
    'scope tracing not supported on torch<2.0')
def test_ram_scope():
    model = DummySequentialForModel()
    input_tensor = torch.randn(1, 3, 32, 32)
    nodes_info = profile_ram(model, input_tensor, detailed=True)
    assert nodes_info.iloc[0].scope == 'layers.0'
    assert get_submodule(model, nodes_info.iloc[0].scope)
    model = DummySequentialSequentialModel()
    nodes_info = profile_ram(model, input_tensor, detailed=True)
    assert nodes_info.iloc[0].scope == 'all_layers.0.0'
    assert get_submodule(model, nodes_info.iloc[0].scope)
    model = DummySequentialSequentialForModel()
    nodes_info = profile_ram(model, input_tensor, detailed=True)
    assert nodes_info.iloc[0].scope == 'all_layers.0.0'
    assert get_submodule(model, nodes_info.iloc[0].scope)
    model = nn.Sequential(nn.Conv2d(3, 16, 1), nn.Conv2d(16, 2, 1))
    nodes_info = profile_ram(model, input_tensor, detailed=True)
    assert nodes_info.iloc[0].scope == '0'
    assert get_submodule(model, nodes_info.iloc[0].scope)
    model = EdgeCaseSequentialModel()
    nodes_info = profile_ram(model, input_tensor, detailed=True)
    with pytest.raises(AssertionError):
        assert nodes_info.iloc[0].scope == 'all_layers.all_layers.0.0'
