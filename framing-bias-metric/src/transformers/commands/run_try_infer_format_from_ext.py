def try_infer_format_from_ext(path: str):
    if not path:
        return 'pipe'
    for ext in PipelineDataFormat.SUPPORTED_FORMATS:
        if path.endswith(ext):
            return ext
    raise Exception(
        'Unable to determine file format from file extension {}. Please provide the format through --format {}'
        .format(path, PipelineDataFormat.SUPPORTED_FORMATS))
