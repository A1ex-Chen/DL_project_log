@pytest.mark.parametrize('mask_value', [100, 200, 300])
def test_mask_persistency(random_tensor, mask_value):
    block = ParallelTransformerBlock(512, 64, 8, 4)
    block.get_mask(mask_value, random_tensor.device)
    assert block.mask.shape[0] == mask_value
