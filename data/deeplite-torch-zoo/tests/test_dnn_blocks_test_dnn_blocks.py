@pytest.mark.parametrize(('block', 'c1', 'c2', 'b', 'res', 'block_kwargs'),
    [(GhostModuleV2, 64, 64, 2, 32, {'dfc': False}), (GhostModuleV2, 64, 64,
    2, 32, {'dfc': True}), (GhostBottleneckV2, 64, 64, 2, 32, {'mid_chs': 
    32}), (MobileOneBlock, 64, 64, 2, 32, {'use_se': True}), (
    MobileOneBlock, 64, 64, 2, 32, {'use_se': False}), (MobileOneBlockUnit,
    64, 64, 2, 32, {'use_se': True}), (MobileOneBlockUnit, 64, 64, 2, 32, {
    'use_se': False}), (FireUnit, 64, 64, 2, 32, {}), (SqnxtUnit, 64, 64, 2,
    32, {}), (RepConv, 64, 64, 2, 32, {}), (MBConv, 64, 64, 2, 32, {}), (
    FusedMBConv, 64, 64, 2, 32, {}), (TwoStackDenseBlock, 64, 64, 2, 32, {}
    ), (RepLKBlock, 64, 64, 2, 32, {}), (RexNetBottleneck, 64, 64, 2, 32, {
    }), (ResNetBasicBlock, 64, 64, 2, 32, {}), (ResNetBottleneck, 64, 64, 2,
    32, {}), (ResNeXtBottleneck, 64, 64, 2, 32, {}), (ShuffleUnit, 64, 64, 
    2, 32, {}), (TransformerBlock, 64, 64, 2, 32, {'num_heads': 2}), (
    SwinTransformerBlock, 64, 64, 2, 32, {'num_heads': 2}), (
    SwinTransformer2Block, 64, 64, 2, 32, {'num_heads': 2}), (STCSPA, 64, 
    64, 2, 32, {'transformer_block': TransformerBlock}), (STCSPB, 64, 64, 2,
    32, {'transformer_block': TransformerBlock}), (STCSPC, 64, 64, 2, 32, {
    'transformer_block': TransformerBlock})])
def test_dnn_blocks(block, c1, c2, b, res, block_kwargs):
    input_tensor = torch.rand((b, c1, res, res), device=None, requires_grad
        =True)
    block = block(c1, c2, **block_kwargs)
    output = block(input_tensor)
    output.sum().backward()
    assert output.shape == (b, c2, res, res)
