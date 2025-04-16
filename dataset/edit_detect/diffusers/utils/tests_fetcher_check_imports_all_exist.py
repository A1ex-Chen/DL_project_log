def check_imports_all_exist():
    """
    Isn't used per se by the test fetcher but might be used later as a quality check. Putting this here for now so the
    code is not lost. This checks all imports in a given file do exist.
    """
    cache = {}
    all_modules = list(PATH_TO_DIFFUSERS.glob('**/*.py')) + list(PATH_TO_TESTS
        .glob('**/*.py'))
    all_modules = [str(mod.relative_to(PATH_TO_REPO)) for mod in all_modules]
    direct_deps = {m: get_module_dependencies(m, cache=cache) for m in
        all_modules}
    for module, deps in direct_deps.items():
        for dep in deps:
            if not (PATH_TO_REPO / dep).is_file():
                print(f'{module} has dependency on {dep} which does not exist.'
                    )
