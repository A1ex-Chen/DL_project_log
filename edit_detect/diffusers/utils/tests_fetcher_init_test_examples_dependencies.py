def init_test_examples_dependencies() ->Tuple[Dict[str, List[str]], List[str]]:
    """
    The test examples do not import from the examples (which are just scripts, not modules) so we need som extra
    care initializing the dependency map, which is the goal of this function. It initializes the dependency map for
    example files by linking each example to the example test file for the example framework.

    Returns:
        `Tuple[Dict[str, List[str]], List[str]]`: A tuple with two elements: the initialized dependency map which is a
        dict test example file to list of example files potentially tested by that test file, and the list of all
        example files (to avoid recomputing it later).
    """
    test_example_deps = {}
    all_examples = []
    for framework in ['flax', 'pytorch', 'tensorflow']:
        test_files = list((PATH_TO_EXAMPLES / framework).glob('test_*.py'))
        all_examples.extend(test_files)
        examples = [f for f in (PATH_TO_EXAMPLES / framework).glob(
            '**/*.py') if f.parent != PATH_TO_EXAMPLES / framework]
        all_examples.extend(examples)
        for test_file in test_files:
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()
            test_example_deps[str(test_file.relative_to(PATH_TO_REPO))] = [str
                (e.relative_to(PATH_TO_REPO)) for e in examples if e.name in
                content]
            test_example_deps[str(test_file.relative_to(PATH_TO_REPO))].append(
                str(test_file.relative_to(PATH_TO_REPO)))
    return test_example_deps, all_examples
