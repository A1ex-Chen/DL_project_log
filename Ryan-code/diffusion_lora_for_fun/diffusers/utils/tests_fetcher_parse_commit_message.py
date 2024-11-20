def parse_commit_message(commit_message: str) ->Dict[str, bool]:
    """
    Parses the commit message to detect if a command is there to skip, force all or part of the CI.

    Args:
        commit_message (`str`): The commit message of the current commit.

    Returns:
        `Dict[str, bool]`: A dictionary of strings to bools with keys the following keys: `"skip"`,
        `"test_all_models"` and `"test_all"`.
    """
    if commit_message is None:
        return {'skip': False, 'no_filter': False, 'test_all': False}
    command_search = re.search('\\[([^\\]]*)\\]', commit_message)
    if command_search is not None:
        command = command_search.groups()[0]
        command = command.lower().replace('-', ' ').replace('_', ' ')
        skip = command in ['ci skip', 'skip ci', 'circleci skip',
            'skip circleci']
        no_filter = set(command.split(' ')) == {'no', 'filter'}
        test_all = set(command.split(' ')) == {'test', 'all'}
        return {'skip': skip, 'no_filter': no_filter, 'test_all': test_all}
    else:
        return {'skip': False, 'no_filter': False, 'test_all': False}
