def set_param(torch_layer, weight, bias=None):
    assert torch_layer.weight.shape == weight.shape, '{} layer.weight does not match'.format(
        torch_layer)
    torch_layer.weight = torch.nn.Parameter(weight)
    if bias is not None:
        assert torch_layer.bias.shape == bias.shape, '{} layer.bias does not match'.format(
            torch_layer)
        torch_layer.bias = torch.nn.Parameter(bias)
