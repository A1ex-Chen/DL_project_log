def is_kaggle():
    """
    Check if the current script is running inside a Kaggle kernel.

    Returns:
        (bool): True if running inside a Kaggle kernel, False otherwise.
    """
    return os.environ.get('PWD') == '/kaggle/working' and os.environ.get(
        'KAGGLE_URL_BASE') == 'https://www.kaggle.com'
