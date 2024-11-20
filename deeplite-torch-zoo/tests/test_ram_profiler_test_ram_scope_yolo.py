@pytest.mark.skipif(not TORCH_2_0, reason=
    'scope tracing not supported on torch<2.0')
def test_ram_scope_yolo():
    model = get_model('yolo5n', 'coco', False)
    input_tensor = torch.randn(1, 3, 96, 96)
    nodes_info = profile_ram(model, input_tensor, detailed=True)
    assert sum(nodes_info['scope'] == 'model.23.cv1.conv') == 1
