def get_class_obj_and_candidates(library_name, class_name,
    importable_classes, pipelines, is_pipeline_module, component_name=None,
    cache_dir=None):
    """Simple helper method to retrieve class object of module as well as potential parent class objects"""
    component_folder = os.path.join(cache_dir, component_name)
    if is_pipeline_module:
        pipeline_module = getattr(pipelines, library_name)
        class_obj = getattr(pipeline_module, class_name)
        class_candidates = {c: class_obj for c in importable_classes.keys()}
    elif os.path.isfile(os.path.join(component_folder, library_name + '.py')):
        class_obj = get_class_from_dynamic_module(component_folder,
            module_file=library_name + '.py', class_name=class_name)
        class_candidates = {c: class_obj for c in importable_classes.keys()}
    else:
        library = importlib.import_module(library_name)
        class_obj = getattr(library, class_name)
        class_candidates = {c: getattr(library, c, None) for c in
            importable_classes.keys()}
    return class_obj, class_candidates
