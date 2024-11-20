def create_reverse_dependency_tree() ->List[Tuple[str, str]]:
    """
    Create a list of all edges (a, b) which mean that modifying a impacts b with a going over all module and test files.
    """
    cache = {}
    all_modules = list(PATH_TO_DIFFUSERS.glob('**/*.py')) + list(PATH_TO_TESTS
        .glob('**/*.py'))
    all_modules = [str(mod.relative_to(PATH_TO_REPO)) for mod in all_modules]
    edges = [(dep, mod) for mod in all_modules for dep in
        get_module_dependencies(mod, cache=cache)]
    return list(set(edges))
