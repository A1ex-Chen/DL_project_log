def get_class_obj_and_candidates(library_name, class_name,
    importable_classes, pipelines, is_pipeline_module):
    """Simple helper method to retrieve class object of module as well as potential parent class objects"""
    if is_pipeline_module:
        pipeline_module = getattr(pipelines, library_name)
        class_obj = getattr(pipeline_module, class_name)
        class_candidates = {c: class_obj for c in importable_classes.keys()}
    else:
        library = importlib.import_module(library_name)
        class_obj = getattr(library, class_name)
        class_candidates = {c: getattr(library, c, None) for c in
            importable_classes.keys()}
    return class_obj, class_candidates
