def create_module_to_test_map(reverse_map: Dict[str, List[str]]=None) ->Dict[
    str, List[str]]:
    """
    Extract the tests from the reverse_dependency_map and potentially filters the model tests.

    Args:
        reverse_map (`Dict[str, List[str]]`, *optional*):
            The reverse dependency map as created by `create_reverse_dependency_map`. Will default to the result of
            that function if not provided.
        filter_pipelines (`bool`, *optional*, defaults to `False`):
            Whether or not to filter pipeline tests to only include core pipelines if a file impacts a lot of models.

    Returns:
        `Dict[str, List[str]]`: A dictionary that maps each file to the tests to execute if that file was modified.
    """
    if reverse_map is None:
        reverse_map = create_reverse_dependency_map()

    def is_test(fname):
        if fname.startswith('tests'):
            return True
        if fname.startswith('examples') and fname.split(os.path.sep)[-1
            ].startswith('test'):
            return True
        return False
    test_map = {module: [f for f in deps if is_test(f)] for module, deps in
        reverse_map.items()}
    return test_map
