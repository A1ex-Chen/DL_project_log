@staticmethod
def load_model(path: Union[str, Path], provider=None, sess_options=None):
    """
        Loads an ONNX Inference session with an ExecutionProvider. Default provider is `CPUExecutionProvider`

        Arguments:
            path (`str` or `Path`):
                Directory from which to load
            provider(`str`, *optional*):
                Onnxruntime execution provider to use for loading the model, defaults to `CPUExecutionProvider`
        """
    if provider is None:
        logger.info(
            'No onnxruntime provider specified, using CPUExecutionProvider')
        provider = 'CPUExecutionProvider'
    return ort.InferenceSession(path, providers=[provider], sess_options=
        sess_options)
