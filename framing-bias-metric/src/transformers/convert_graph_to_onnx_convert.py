def convert(framework: str, model: str, output: Path, opset: int, tokenizer:
    Optional[str]=None, use_external_format: bool=False, pipeline_name: str
    ='feature-extraction'):
    """
    Convert the pipeline object to the ONNX Intermediate Representation (IR) format

    Args:
        framework: The framework the pipeline is backed by ("pt" or "tf")
        model: The name of the model to load for the pipeline
        output: The path where the ONNX graph will be stored
        opset: The actual version of the ONNX operator set to use
        tokenizer: The name of the model to load for the pipeline, default to the model's name if not provided
        use_external_format: Split the model definition from its parameters to allow model bigger than 2GB (PyTorch only)
        pipeline_name: The kind of pipeline to instantiate (ner, question-answering, etc.)

    Returns:

    """
    print(f'ONNX opset version set to: {opset}')
    nlp = load_graph_from_args(pipeline_name, framework, model, tokenizer)
    if not output.parent.exists():
        print(f'Creating folder {output.parent}')
        makedirs(output.parent.as_posix())
    elif len(listdir(output.parent.as_posix())) > 0:
        raise Exception(
            f'Folder {output.parent.as_posix()} is not empty, aborting conversion'
            )
    if framework == 'pt':
        convert_pytorch(nlp, opset, output, use_external_format)
    else:
        convert_tensorflow(nlp, opset, output)
