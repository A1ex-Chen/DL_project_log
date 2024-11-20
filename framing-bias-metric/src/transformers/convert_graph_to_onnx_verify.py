def verify(path: Path):
    from onnxruntime import InferenceSession, SessionOptions
    from onnxruntime.capi.onnxruntime_pybind11_state import RuntimeException
    print(f'Checking ONNX model loading from: {path} ...')
    try:
        onnx_options = SessionOptions()
        _ = InferenceSession(path.as_posix(), onnx_options, providers=[
            'CPUExecutionProvider'])
        print(f'Model {path} correctly loaded: ✔')
    except RuntimeException as re:
        print(f'Error while loading the model {re}: ✘')
