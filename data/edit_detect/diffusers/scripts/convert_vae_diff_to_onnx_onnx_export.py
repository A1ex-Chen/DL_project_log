def onnx_export(model, model_args: tuple, output_path: Path,
    ordered_input_names, output_names, dynamic_axes, opset,
    use_external_data_format=False):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if is_torch_less_than_1_11:
        export(model, model_args, f=output_path.as_posix(), input_names=
            ordered_input_names, output_names=output_names, dynamic_axes=
            dynamic_axes, do_constant_folding=True,
            use_external_data_format=use_external_data_format,
            enable_onnx_checker=True, opset_version=opset)
    else:
        export(model, model_args, f=output_path.as_posix(), input_names=
            ordered_input_names, output_names=output_names, dynamic_axes=
            dynamic_axes, do_constant_folding=True, opset_version=opset)
