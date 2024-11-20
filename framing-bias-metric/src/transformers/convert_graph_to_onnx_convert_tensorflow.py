def convert_tensorflow(nlp: Pipeline, opset: int, output: Path):
    """
    Export a TensorFlow backed pipeline to ONNX Intermediate Representation (IR

    Args:
        nlp: The pipeline to be exported
        opset: The actual version of the ONNX operator set to use
        output: Path where will be stored the generated ONNX model

    Notes: TensorFlow cannot export model bigger than 2GB due to internal constraint from TensorFlow

    """
    if not is_tf_available():
        raise Exception(
            'Cannot convert because TF is not installed. Please install tensorflow first.'
            )
    print(
        "/!\\ Please note TensorFlow doesn't support exporting model > 2Gb /!\\"
        )
    try:
        import tensorflow as tf
        from keras2onnx import __version__ as k2ov
        from keras2onnx import convert_keras, save_model
        print(
            f'Using framework TensorFlow: {tf.version.VERSION}, keras2onnx: {k2ov}'
            )
        input_names, output_names, dynamic_axes, tokens = infer_shapes(nlp,
            'tf')
        nlp.model.predict(tokens.data)
        onnx_model = convert_keras(nlp.model, nlp.model.name, target_opset=
            opset)
        save_model(onnx_model, output.as_posix())
    except ImportError as e:
        raise Exception(
            f'Cannot import {e.name} required to convert TF model to ONNX. Please install {e.name} first.'
            )
