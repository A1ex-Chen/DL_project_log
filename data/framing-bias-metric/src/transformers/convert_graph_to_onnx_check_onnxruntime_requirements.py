def check_onnxruntime_requirements(minimum_version: Version):
    """
    Check onnxruntime is installed and if the installed version match is recent enough

    Raises:
        ImportError: If onnxruntime is not installed or too old version is found
    """
    try:
        import onnxruntime
        ort_version = parse(onnxruntime.__version__)
        if ort_version < ORT_QUANTIZE_MINIMUM_VERSION:
            raise ImportError(
                f"""We found an older version of onnxruntime ({onnxruntime.__version__}) but we require onnxruntime to be >= {minimum_version} to enable all the conversions options.
Please update onnxruntime by running `pip install --upgrade onnxruntime`"""
                )
    except ImportError:
        raise ImportError(
            "onnxruntime doesn't seem to be currently installed. Please install the onnxruntime by running `pip install onnxruntime` and relaunch the conversion."
            )
