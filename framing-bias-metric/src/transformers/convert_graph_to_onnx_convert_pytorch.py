def convert_pytorch(nlp: Pipeline, opset: int, output: Path,
    use_external_format: bool):
    """
    Export a PyTorch backed pipeline to ONNX Intermediate Representation (IR

    Args:
        nlp: The pipeline to be exported
        opset: The actual version of the ONNX operator set to use
        output: Path where will be stored the generated ONNX model
        use_external_format: Split the model definition from its parameters to allow model bigger than 2GB

    Returns:

    """
    if not is_torch_available():
        raise Exception(
            'Cannot convert because PyTorch is not installed. Please install torch first.'
            )
    import torch
    from torch.onnx import export
    print(f'Using framework PyTorch: {torch.__version__}')
    with torch.no_grad():
        input_names, output_names, dynamic_axes, tokens = infer_shapes(nlp,
            'pt')
        ordered_input_names, model_args = ensure_valid_input(nlp.model,
            tokens, input_names)
        export(nlp.model, model_args, f=output.as_posix(), input_names=
            ordered_input_names, output_names=output_names, dynamic_axes=
            dynamic_axes, do_constant_folding=True,
            use_external_data_format=use_external_format,
            enable_onnx_checker=True, opset_version=opset)
