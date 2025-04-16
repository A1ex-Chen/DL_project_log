def export_onnx_model(model, inputs):
    """
    Trace and export a model to onnx format.

    Args:
        model (nn.Module):
        inputs (tuple[args]): the model will be called by `model(*inputs)`

    Returns:
        an onnx model
    """
    assert isinstance(model, torch.nn.Module)

    def _check_eval(module):
        assert not module.training
    model.apply(_check_eval)
    with torch.no_grad():
        with io.BytesIO() as f:
            torch.onnx.export(model, inputs, f, operator_export_type=
                OperatorExportTypes.ONNX_ATEN_FALLBACK)
            onnx_model = onnx.load_from_string(f.getvalue())
    all_passes = onnx.optimizer.get_available_passes()
    passes = ['fuse_bn_into_conv']
    assert all(p in all_passes for p in passes)
    onnx_model = onnx.optimizer.optimize(onnx_model, passes)
    return onnx_model
