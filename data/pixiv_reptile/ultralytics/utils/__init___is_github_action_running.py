def is_github_action_running() ->bool:
    """
    Determine if the current environment is a GitHub Actions runner.

    Returns:
        (bool): True if the current environment is a GitHub Actions runner, False otherwise.
    """
    return ('GITHUB_ACTIONS' in os.environ and 'GITHUB_WORKFLOW' in os.
        environ and 'RUNNER_OS' in os.environ)
