def is_interactive() ->bool:
    """True if running in a interactive environment/jupyter notebook.

    Returns:
        bool: True if running in an interactive environment
    """
    return not hasattr(main, '__file__')
