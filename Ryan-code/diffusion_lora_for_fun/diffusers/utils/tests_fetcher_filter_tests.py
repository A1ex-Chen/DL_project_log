def filter_tests(output_file: str, filters: List[str]):
    """
    Reads the content of the output file and filters out all the tests in a list of given folders.

    Args:
        output_file (`str` or `os.PathLike`): The path to the output file of the tests fetcher.
        filters (`List[str]`): A list of folders to filter.
    """
    if not os.path.isfile(output_file):
        print('No test file found.')
        return
    with open(output_file, 'r', encoding='utf-8') as f:
        test_files = f.read().split(' ')
    if len(test_files) == 0 or test_files == ['']:
        print('No tests to filter.')
        return
    if test_files == ['tests']:
        test_files = [os.path.join('tests', f) for f in os.listdir('tests') if
            f not in ['__init__.py'] + filters]
    else:
        test_files = [f for f in test_files if f.split(os.path.sep)[1] not in
            filters]
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(' '.join(test_files))
