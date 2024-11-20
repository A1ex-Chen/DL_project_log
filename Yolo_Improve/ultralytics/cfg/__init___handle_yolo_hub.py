def handle_yolo_hub(args: List[str]) ->None:
    """
    Handle Ultralytics HUB command-line interface (CLI) commands.

    This function processes Ultralytics HUB CLI commands such as login and logout. It should be called when executing
    a script with arguments related to HUB authentication.

    Args:
        args (List[str]): A list of command line arguments.

    Returns:
        None

    Example:
        ```bash
        yolo hub login YOUR_API_KEY
        ```
    """
    from ultralytics import hub
    if args[0] == 'login':
        key = args[1] if len(args) > 1 else ''
        hub.login(key)
    elif args[0] == 'logout':
        hub.logout()
