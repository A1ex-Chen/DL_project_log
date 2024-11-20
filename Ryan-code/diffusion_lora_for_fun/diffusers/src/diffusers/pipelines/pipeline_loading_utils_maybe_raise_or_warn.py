def maybe_raise_or_warn(library_name, library, class_name,
    importable_classes, passed_class_obj, name, is_pipeline_module):
    """Simple helper method to raise or warn in case incorrect module has been passed"""
    if not is_pipeline_module:
        library = importlib.import_module(library_name)
        class_obj = getattr(library, class_name)
        class_candidates = {c: getattr(library, c, None) for c in
            importable_classes.keys()}
        expected_class_obj = None
        for class_name, class_candidate in class_candidates.items():
            if class_candidate is not None and issubclass(class_obj,
                class_candidate):
                expected_class_obj = class_candidate
        sub_model = passed_class_obj[name]
        unwrapped_sub_model = _unwrap_model(sub_model)
        model_cls = unwrapped_sub_model.__class__
        if not issubclass(model_cls, expected_class_obj):
            raise ValueError(
                f'{passed_class_obj[name]} is of type: {model_cls}, but should be {expected_class_obj}'
                )
    else:
        logger.warning(
            f'You have passed a non-standard module {passed_class_obj[name]}. We cannot verify whether it has the correct type'
            )
