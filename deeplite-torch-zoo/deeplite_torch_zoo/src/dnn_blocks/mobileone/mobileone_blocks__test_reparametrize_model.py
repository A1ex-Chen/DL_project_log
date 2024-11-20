def _test_reparametrize_model(c1, c2, b=2, res=32):
    input = torch.rand((b, c1, res, res), device=None, requires_grad=False)
    block = MobileOneBlock(c1, c2, 3)
    weight_count1 = _calc_width(block)
    block = fuse_blocks(block)
    weight_count2 = _calc_width(block)
    output = block(input)
    assert weight_count1 != weight_count2
    LOGGER.info(
        f'Weight Before and after reparameterization {weight_count1} -> {weight_count2}'
        )
    block = MobileOneBlockUnit(c1, c2, 3)
    weight_count1 = _calc_width(block)
    block = fuse_blocks(block)
    weight_count2 = _calc_width(block)
    output = block(input)
    assert weight_count1 != weight_count2
    LOGGER.info(
        f'Weight Before and after reparameterization {weight_count1} -> {weight_count2}'
        )
