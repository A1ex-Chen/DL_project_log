def create_reverse_dependency_map() ->Dict[str, List[str]]:
    """
    Create the dependency map from module/test filename to the list of modules/tests that depend on it recursively.

    Returns:
        `Dict[str, List[str]]`: The reverse dependency map as a dictionary mapping filenames to all the filenames
        depending on it recursively. This way the tests impacted by a change in file A are the test files in the list
        corresponding to key A in this result.
    """
    cache = {}
    example_deps, examples = init_test_examples_dependencies()
    all_modules = list(PATH_TO_DIFFUSERS.glob('**/*.py')) + list(PATH_TO_TESTS
        .glob('**/*.py')) + examples
    all_modules = [str(mod.relative_to(PATH_TO_REPO)) for mod in all_modules]
    direct_deps = {m: get_module_dependencies(m, cache=cache) for m in
        all_modules}
    direct_deps.update(example_deps)
    something_changed = True
    while something_changed:
        something_changed = False
        for m in all_modules:
            for d in direct_deps[m]:
                if d.endswith('__init__.py'):
                    continue
                if d not in direct_deps:
                    raise ValueError(f'KeyError:{d}. From {m}')
                new_deps = set(direct_deps[d]) - set(direct_deps[m])
                if len(new_deps) > 0:
                    direct_deps[m].extend(list(new_deps))
                    something_changed = True
    reverse_map = collections.defaultdict(list)
    for m in all_modules:
        for d in direct_deps[m]:
            reverse_map[d].append(m)
    for m in [f for f in all_modules if f.endswith('__init__.py')]:
        direct_deps = get_module_dependencies(m, cache=cache)
        deps = sum([reverse_map[d] for d in direct_deps if not d.endswith(
            '__init__.py')], direct_deps)
        reverse_map[m] = list(set(deps) - {m})
    return reverse_map
