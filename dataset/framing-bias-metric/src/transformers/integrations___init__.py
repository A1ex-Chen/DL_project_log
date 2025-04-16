def __init__(self):
    assert _has_mlflow, 'MLflowCallback requires mlflow to be installed. Run `pip install mlflow`.'
    self._initialized = False
    self._log_artifacts = False
