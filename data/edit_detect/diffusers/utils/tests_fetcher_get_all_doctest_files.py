def get_all_doctest_files() ->List[str]:
    """
    Return the complete list of python and Markdown files on which we run doctest.

    At this moment, we restrict this to only take files from `src/` or `docs/source/en/` that are not in `utils/not_doctested.txt`.

    Returns:
        `List[str]`: The complete list of Python and Markdown files on which we run doctest.
    """
    py_files = [str(x.relative_to(PATH_TO_REPO)) for x in PATH_TO_REPO.glob
        ('**/*.py')]
    md_files = [str(x.relative_to(PATH_TO_REPO)) for x in PATH_TO_REPO.glob
        ('**/*.md')]
    test_files_to_run = py_files + md_files
    test_files_to_run = [x for x in test_files_to_run if x.startswith((
        'src/', 'docs/source/en/'))]
    test_files_to_run = [x for x in test_files_to_run if not x.endswith((
        '__init__.py',))]
    with open('utils/not_doctested.txt') as fp:
        not_doctested = {x.split(' ')[0] for x in fp.read().strip().split('\n')
            }
    test_files_to_run = [x for x in test_files_to_run if x not in not_doctested
        ]
    return sorted(test_files_to_run)
