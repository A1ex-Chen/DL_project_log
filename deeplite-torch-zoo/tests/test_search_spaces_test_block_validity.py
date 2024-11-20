@pytest.mark.parametrize(('model_name', 'input_shape', 'block_key'),
    BLOCK_VALIDITY_TESTS)
def test_block_validity(model_name, input_shape, block_key):
    model = get_model_by_name(model_name=model_name, dataset_name=
        'imagenet', device='cpu')
    block_cls = block_registry.get(block_key)
    model.features.stage1.unit2 = block_cls(64, 64)
    y = model(torch.randn(*input_shape))
    y.sum().backward()
