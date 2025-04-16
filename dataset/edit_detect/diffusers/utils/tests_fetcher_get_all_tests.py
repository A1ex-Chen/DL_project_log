def get_all_tests() ->List[str]:
    """
    Walks the `tests` folder to return a list of files/subfolders. This is used to split the tests to run when using
    paralellism. The split is:

    - folders under `tests`: (`tokenization`, `pipelines`, etc) except the subfolder `models` is excluded.
    - folders under `tests/models`: `bert`, `gpt2`, etc.
    - test files under `tests`: `test_modeling_common.py`, `test_tokenization_common.py`, etc.
    """
    tests = os.listdir(PATH_TO_TESTS)
    tests = [f'tests/{f}' for f in tests if '__pycache__' not in f]
    tests = sorted([f for f in tests if (PATH_TO_REPO / f).is_dir() or f.
        startswith('tests/test_')])
    return tests
