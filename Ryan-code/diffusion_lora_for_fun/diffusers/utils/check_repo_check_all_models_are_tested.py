def check_all_models_are_tested():
    """Check all models are properly tested."""
    modules = get_model_modules()
    test_files = get_model_test_files()
    failures = []
    for module in modules:
        test_file = [file for file in test_files if 
            f"test_{module.__name__.split('.')[-1]}.py" in file]
        if len(test_file) == 0:
            failures.append(
                f'{module.__name__} does not have its corresponding test file {test_file}.'
                )
        elif len(test_file) > 1:
            failures.append(
                f'{module.__name__} has several test files: {test_file}.')
        else:
            test_file = test_file[0]
            new_failures = check_models_are_tested(module, test_file)
            if new_failures is not None:
                failures += new_failures
    if len(failures) > 0:
        raise Exception(f'There were {len(failures)} failures:\n' + '\n'.
            join(failures))
