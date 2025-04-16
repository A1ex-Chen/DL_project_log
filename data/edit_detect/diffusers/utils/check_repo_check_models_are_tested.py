def check_models_are_tested(module, test_file):
    """Check models defined in module are tested in test_file."""
    defined_models = get_models(module)
    tested_models = find_tested_models(test_file)
    if tested_models is None:
        if test_file.replace(os.path.sep, '/'
            ) in TEST_FILES_WITH_NO_COMMON_TESTS:
            return
        return [
            f'{test_file} should define `all_model_classes` to apply common tests to the models it tests. '
             +
            'If this intentional, add the test filename to `TEST_FILES_WITH_NO_COMMON_TESTS` in the file '
             + '`utils/check_repo.py`.']
    failures = []
    for model_name, _ in defined_models:
        if (model_name not in tested_models and model_name not in
            IGNORE_NON_TESTED):
            failures.append(
                f'{model_name} is defined in {module.__name__} but is not tested in '
                 +
                f'{os.path.join(PATH_TO_TESTS, test_file)}. Add it to the all_model_classes in that file.'
                 +
                'If common tests should not applied to that model, add its name to `IGNORE_NON_TESTED`'
                 + 'in the file `utils/check_repo.py`.')
    return failures
