def alias(x, name, is_backward=False):
    if not torch.onnx.is_in_onnx_export():
        return x
    assert isinstance(x, torch.Tensor)
    return torch.ops._caffe2.AliasWithName(x, name, is_backward=is_backward)
