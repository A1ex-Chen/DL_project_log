def optimize(onnx_model_path: Path) ->Path:
    """
    Load the model at the specified path and let onnxruntime look at transformations on the graph to enable all the
    optimizations possibl

    Args:
        onnx_model_path: filepath where the model binary description is stored

    Returns: Path where the optimized model binary description has been saved

    """
    from onnxruntime import InferenceSession, SessionOptions
    opt_model_path = generate_identified_filename(onnx_model_path, '-optimized'
        )
    sess_option = SessionOptions()
    sess_option.optimized_model_filepath = opt_model_path.as_posix()
    _ = InferenceSession(onnx_model_path.as_posix(), sess_option)
    print(f'Optimized model has been written at {opt_model_path}: ✔')
    print(
        '/!\\ Optimized model contains hardware specific operators which might not be portable. /!\\'
        )
    return opt_model_path
