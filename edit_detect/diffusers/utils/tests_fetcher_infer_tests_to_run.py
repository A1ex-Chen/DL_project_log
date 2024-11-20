def infer_tests_to_run(output_file: str, diff_with_last_commit: bool=False,
    json_output_file: Optional[str]=None):
    """
    The main function called by the test fetcher. Determines the tests to run from the diff.

    Args:
        output_file (`str`):
            The path where to store the summary of the test fetcher analysis. Other files will be stored in the same
            folder:

            - examples_test_list.txt: The list of examples tests to run.
            - test_repo_utils.txt: Will indicate if the repo utils tests should be run or not.
            - doctest_list.txt: The list of doctests to run.

        diff_with_last_commit (`bool`, *optional*, defaults to `False`):
            Whether to analyze the diff with the last commit (for use on the main branch after a PR is merged) or with
            the branching point from main (for use on each PR).
        filter_models (`bool`, *optional*, defaults to `True`):
            Whether or not to filter the tests to core models only, when a file modified results in a lot of model
            tests.
        json_output_file (`str`, *optional*):
            The path where to store the json file mapping categories of tests to tests to run (used for parallelism or
            the slow tests).
    """
    modified_files = get_modified_python_files(diff_with_last_commit=
        diff_with_last_commit)
    print(f'\n### MODIFIED FILES ###\n{_print_list(modified_files)}')
    reverse_map = create_reverse_dependency_map()
    impacted_files = modified_files.copy()
    for f in modified_files:
        if f in reverse_map:
            impacted_files.extend(reverse_map[f])
    impacted_files = sorted(set(impacted_files))
    print(f'\n### IMPACTED FILES ###\n{_print_list(impacted_files)}')
    if any(x in modified_files for x in ['setup.py']):
        test_files_to_run = ['tests', 'examples']
    if 'tests/utils/tiny_model_summary.json' in modified_files:
        test_files_to_run = ['tests']
        any(f.split(os.path.sep)[0] == 'utils' for f in modified_files)
    else:
        test_files_to_run = [f for f in modified_files if f.startswith(
            'tests') and f.split(os.path.sep)[-1].startswith('test')]
        test_map = create_module_to_test_map(reverse_map=reverse_map)
        for f in modified_files:
            if f in test_map:
                test_files_to_run.extend(test_map[f])
        test_files_to_run = sorted(set(test_files_to_run))
        test_files_to_run = [f for f in test_files_to_run if (PATH_TO_REPO /
            f).exists()]
        any(f.split(os.path.sep)[0] == 'utils' for f in modified_files)
    examples_tests_to_run = [f for f in test_files_to_run if f.startswith(
        'examples')]
    test_files_to_run = [f for f in test_files_to_run if not f.startswith(
        'examples')]
    print(f'\n### TEST TO RUN ###\n{_print_list(test_files_to_run)}')
    if len(test_files_to_run) > 0:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(' '.join(test_files_to_run))
        if 'tests' in test_files_to_run:
            test_files_to_run = get_all_tests()
        create_json_map(test_files_to_run, json_output_file)
    print(
        f'\n### EXAMPLES TEST TO RUN ###\n{_print_list(examples_tests_to_run)}'
        )
    if len(examples_tests_to_run) > 0:
        if examples_tests_to_run == ['examples']:
            examples_tests_to_run = ['all']
        example_file = Path(output_file).parent / 'examples_test_list.txt'
        with open(example_file, 'w', encoding='utf-8') as f:
            f.write(' '.join(examples_tests_to_run))
