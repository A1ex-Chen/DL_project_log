def check_all_objects_are_documented():
    """Check all models are properly documented."""
    documented_objs = find_all_documented_objects()
    modules = diffusers._modules
    objects = [c for c in dir(diffusers) if c not in modules and not c.
        startswith('_')]
    undocumented_objs = [c for c in objects if c not in documented_objs and
        not ignore_undocumented(c)]
    if len(undocumented_objs) > 0:
        raise Exception(
            """The following objects are in the public init so should be documented:
 - """
             + '\n - '.join(undocumented_objs))
    check_docstrings_are_in_md()
    check_model_type_doc_match()
