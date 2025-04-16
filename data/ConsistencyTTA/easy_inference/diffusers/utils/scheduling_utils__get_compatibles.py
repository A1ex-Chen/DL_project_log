@classmethod
def _get_compatibles(cls):
    compatible_classes_str = list(set([cls.__name__] + cls._compatibles))
    diffusers_library = importlib.import_module(__name__.split('.')[0])
    compatible_classes = [getattr(diffusers_library, c) for c in
        compatible_classes_str if hasattr(diffusers_library, c)]
    return compatible_classes
