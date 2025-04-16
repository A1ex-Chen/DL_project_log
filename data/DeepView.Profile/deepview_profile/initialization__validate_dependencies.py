def _validate_dependencies():
    try:
        import yaml
        import pynvml
        import google.protobuf
        import numpy
        import torch
        return True
    except ImportError as ex:
        logger.error(
            "DeepView could not find the '%s' module, which is a required dependency. Please make sure all the required dependencies are installed before launching DeepView. If you use a package manager, these dependencies will be automatically installed for you."
            , ex.name)
        return False
