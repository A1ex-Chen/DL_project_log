def set_sentry():
    """
    Initialize the Sentry SDK for error tracking and reporting. Only used if sentry_sdk package is installed and
    sync=True in settings. Run 'yolo settings' to see and update settings YAML file.

    Conditions required to send errors (ALL conditions must be met or no errors will be reported):
        - sentry_sdk package is installed
        - sync=True in YOLO settings
        - pytest is not running
        - running in a pip package installation
        - running in a non-git directory
        - running with rank -1 or 0
        - online environment
        - CLI used to run package (checked with 'yolo' as the name of the main CLI command)

    The function also configures Sentry SDK to ignore KeyboardInterrupt and FileNotFoundError
    exceptions and to exclude events with 'out of memory' in their exception message.

    Additionally, the function sets custom tags and user information for Sentry events.
    """

    def before_send(event, hint):
        """
        Modify the event before sending it to Sentry based on specific exception types and messages.

        Args:
            event (dict): The event dictionary containing information about the error.
            hint (dict): A dictionary containing additional information about the error.

        Returns:
            dict: The modified event or None if the event should not be sent to Sentry.
        """
        if 'exc_info' in hint:
            exc_type, exc_value, tb = hint['exc_info']
            if exc_type in {KeyboardInterrupt, FileNotFoundError
                } or 'out of memory' in str(exc_value):
                return None
        event['tags'] = {'sys_argv': ARGV[0], 'sys_argv_name': Path(ARGV[0]
            ).name, 'install': 'git' if IS_GIT_DIR else 'pip' if
            IS_PIP_PACKAGE else 'other', 'os': ENVIRONMENT}
        return event
    if (SETTINGS['sync'] and RANK in {-1, 0} and Path(ARGV[0]).name ==
        'yolo' and not TESTS_RUNNING and ONLINE and IS_PIP_PACKAGE and not
        IS_GIT_DIR):
        try:
            import sentry_sdk
        except ImportError:
            return
        sentry_sdk.init(dsn=
            'https://5ff1556b71594bfea135ff0203a0d290@o4504521589325824.ingest.sentry.io/4504521592406016'
            , debug=False, traces_sample_rate=1.0, release=__version__,
            environment='production', before_send=before_send,
            ignore_errors=[KeyboardInterrupt, FileNotFoundError])
        sentry_sdk.set_user({'id': SETTINGS['uuid']})
