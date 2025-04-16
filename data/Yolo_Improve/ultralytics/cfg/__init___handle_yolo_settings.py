def handle_yolo_settings(args: List[str]) ->None:
    """
    Handle YOLO settings command-line interface (CLI) commands.

    This function processes YOLO settings CLI commands such as reset. It should be called when executing a script with
    arguments related to YOLO settings management.

    Args:
        args (List[str]): A list of command line arguments for YOLO settings management.

    Returns:
        None

    Example:
        ```bash
        yolo settings reset
        ```

    Notes:
        For more information on handling YOLO settings, visit:
        https://docs.ultralytics.com/quickstart/#ultralytics-settings
    """
    url = 'https://docs.ultralytics.com/quickstart/#ultralytics-settings'
    try:
        if any(args):
            if args[0] == 'reset':
                SETTINGS_YAML.unlink()
                SETTINGS.reset()
                LOGGER.info('Settings reset successfully')
            else:
                new = dict(parse_key_value_pair(a) for a in args)
                check_dict_alignment(SETTINGS, new)
                SETTINGS.update(new)
        LOGGER.info(f'💡 Learn about settings at {url}')
        yaml_print(SETTINGS_YAML)
    except Exception as e:
        LOGGER.warning(
            f"WARNING ⚠️ settings error: '{e}'. Please see {url} for help.")
