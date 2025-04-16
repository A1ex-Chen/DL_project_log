@pytest.mark.parametrize('input_value, expected_output_shape', [(torch.
    randint(0, 20000, (1, 512)), (1, 20000)), (torch.randint(0, 20000, (32,
    256)), (32, 20000))])
def test_bitnet_transformer_output_shapes(bitnet_model, input_value,
    expected_output_shape):
    logits = bitnet_model(input_value)
    assert logits.shape == expected_output_shape
