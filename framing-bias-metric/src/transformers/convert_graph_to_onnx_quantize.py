def quantize(onnx_model_path: Path) ->Path:
    """
    Quantize the weights of the model from float32 to in8 to allow very efficient inference on modern CPU

    Args:
        onnx_model_path: Path to location the exported ONNX model is stored

    Returns: The Path generated for the quantized
    """
    import onnx
    from onnxruntime.quantization import QuantizationMode, quantize
    onnx_model = onnx.load(onnx_model_path.as_posix())
    print(
        """As of onnxruntime 1.4.0, models larger than 2GB will fail to quantize due to protobuf constraint.
This limitation will be removed in the next release of onnxruntime."""
        )
    quantized_model = quantize(model=onnx_model, quantization_mode=
        QuantizationMode.IntegerOps, force_fusions=True, symmetric_weight=True)
    quantized_model_path = generate_identified_filename(onnx_model_path,
        '-quantized')
    print(f'Quantized model has been written at {quantized_model_path}: ✔')
    onnx.save_model(quantized_model, quantized_model_path.as_posix())
    return quantized_model_path
