def _log_api_usage(identifier: str):
    """
    Internal function used to log the usage of different detectron2 components
    inside facebook's infra.
    """
    torch._C._log_api_usage_once('detectron2.' + identifier)
