def create_json_map(test_files_to_run: List[str], json_output_file:
    Optional[str]=None):
    """
    Creates a map from a list of tests to run to easily split them by category, when running parallelism of slow tests.

    Args:
        test_files_to_run (`List[str]`): The list of tests to run.
        json_output_file (`str`): The path where to store the built json map.
    """
    if json_output_file is None:
        return
    test_map = {}
    for test_file in test_files_to_run:
        names = test_file.split(os.path.sep)
        module = names[1]
        if module in MODULES_TO_IGNORE:
            continue
        if len(names) > 2 or not test_file.endswith('.py'):
            key = os.path.sep.join(names[1:2])
        else:
            key = 'common'
        if key not in test_map:
            test_map[key] = []
        test_map[key].append(test_file)
    keys = sorted(test_map.keys())
    test_map = {k: ' '.join(sorted(test_map[k])) for k in keys}
    with open(json_output_file, 'w', encoding='UTF-8') as fp:
        json.dump(test_map, fp, ensure_ascii=False)
